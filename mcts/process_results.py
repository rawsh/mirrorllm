import json
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional, Any
from pathlib import Path
import numpy as np

# Changes to PathMetrics class definition:
@dataclass
class PathMetrics:
    """Metrics for a single reasoning path"""
    answer: str
    is_correct: bool
    path_length: int
    prm_score: float
    raw_steps: List[str]  # Store original steps
    steps: List[str]  # Store preprocessed steps

@dataclass
class QuestionAnalysis:
    """Analysis results for a single question"""
    question_text: str
    correct_answer: str
    binary_success: bool  # Any correct path?
    sc_score: float  # Overall self-consistency score
    sc_correct_percent: float  # % of self-consistent answers that are correct
    sc_most_common_correct: bool  # Whether most common answer is correct
    total_paths: int
    correct_paths: List[PathMetrics]
    incorrect_paths: List[PathMetrics]
    answer_distribution: Counter

class MathReasoningAnalyzer:
    def __init__(self, path_data: str):
        self.data_path = Path(path_data)
        self.questions = self._load_data()
        
    def _load_data(self) -> List[dict]:
        """Load and parse JSONL data"""
        questions = []
        with open(self.data_path) as f:
            for line in f:
                if line.strip():
                    questions.append(json.loads(line))
        return questions

    def _extract_steps(self, final_state: str) -> Tuple[List[str], List[str]]:
        """Extract reasoning steps from final state and return both raw and processed steps"""
        raw_steps = [step.strip() for step in final_state.split('\n\n')]
        raw_steps = [step for step in raw_steps if step]
        
        if len(raw_steps) > 2:
            # Remove first step (question repeat)
            processed_steps = raw_steps[1:]
            # Concatenate last formatting step to the previous step
            if len(processed_steps) > 1:
                processed_steps[-2] = processed_steps[-2] + "\n\n" + processed_steps[-1]
                processed_steps = processed_steps[:-1]
        else:
            processed_steps = []
        
        return raw_steps, processed_steps

    def _extract_answer(self, final_state: str) -> str:
        """Extract final answer from the final state"""
        if '\\boxed{' in final_state:
            return final_state.split('\\boxed{')[1].split('}')[0]
        return ''

    def analyze_question(self, question: dict) -> QuestionAnalysis:
        """Analyze a single question's reasoning paths"""
        paths = []
        answers = Counter()
        
        # Process each terminal path
        for path in question['terminal_paths']:
            raw_steps, processed_steps = self._extract_steps(path['final_state'])
            answer = self._extract_answer(path['final_state'])
            
            path_metrics = PathMetrics(
                answer=answer,
                is_correct=path['correct'],
                path_length=len(processed_steps),
                prm_score=path['score'],
                raw_steps=raw_steps,
                steps=processed_steps
            )
            paths.append(path_metrics)
            answers[answer] += 1

        # Split paths
        correct_paths = [p for p in paths if p.is_correct]
        incorrect_paths = [p for p in paths if not p.is_correct]
        
        # Calculate self-consistency and SC correct %
        total_paths = len(paths)
        if total_paths > 0:
            # Overall self-consistency
            most_common = answers.most_common()
            most_common_count = most_common[0][1] if most_common else 0
            most_common_answer = most_common[0][0] if most_common else ''
            sc_score = most_common_count / total_paths
            
            # Calculate % of self-consistent answers that are correct
            sc_answers = [ans for ans, count in most_common 
                        if count > total_paths * 0.2]  # Consider answers that appear >20% of time
            sc_correct = sum(1 for ans in sc_answers 
                        if any(p.answer == ans and p.is_correct 
                                for p in correct_paths))
            sc_correct_percent = sc_correct / len(sc_answers) if sc_answers else 0
            
            # Check if most common answer is correct
            sc_most_common_correct = any(p.answer == most_common_answer and p.is_correct 
                                    for p in correct_paths)
        else:
            sc_score = 0
            sc_correct_percent = 0
            sc_most_common_correct = False

        return QuestionAnalysis(
            question_text=question['question'],
            correct_answer=question['correct_answer'],
            binary_success=bool(correct_paths),
            sc_score=sc_score,
            sc_correct_percent=sc_correct_percent,
            sc_most_common_correct=sc_most_common_correct,
            total_paths=total_paths,
            correct_paths=correct_paths,
            incorrect_paths=incorrect_paths,
            answer_distribution=answers
        )

    # New version of get_paired_examples:
    def get_paired_examples(
        self,
        analyses: List[QuestionAnalysis],
        max_pairs: int = 10000,
        top_n_correct: int = 50  # New parameter
    ) -> List[Dict[str, Any]]:
        """Get paired examples considering multiple correct paths per question"""
        paired_examples = []
        
        for analysis in analyses:
            if not analysis.correct_paths or not analysis.incorrect_paths:
                continue
                
            # Sort correct paths by quality (shorter length + higher PRM score)
            sorted_correct = sorted(
                analysis.correct_paths,
                key=lambda p: (-p.prm_score, p.path_length)
            )
            
            # Take top N correct paths
            top_correct_paths = sorted_correct[:top_n_correct]
            
            # Get shortest correct path length for reference
            shortest_correct_len = min(p.path_length for p in analysis.correct_paths)
            
            # Filter correct paths that aren't too much longer than shortest
            filtered_correct = [
                p for p in top_correct_paths 
                if p.path_length <= shortest_correct_len * 1.4
            ]
            
            # For each correct path, find the most deceptive incorrect path
            for correct_path in filtered_correct:
                # Find most deceptive incorrect path relative to this correct path
                best_incorrect = max(
                    analysis.incorrect_paths,
                    key=lambda p: (
                        p.prm_score,
                        -abs(p.path_length - correct_path.path_length)
                    )
                )
                
                paired_examples.append({
                    'question': analysis.question_text,
                    'correct_answer': analysis.correct_answer,
                    'metrics': {
                        'sc_score': analysis.sc_score,
                        'sc_correct_percent': analysis.sc_correct_percent,
                        'total_paths': analysis.total_paths
                    },
                    'positive': {
                        'steps': correct_path.steps,
                        'answer': correct_path.answer,
                        'prm_score': correct_path.prm_score,
                        'path_length': correct_path.path_length
                    },
                    'negative': {
                        'steps': best_incorrect.steps,
                        'answer': best_incorrect.answer,
                        'prm_score': best_incorrect.prm_score,
                        'path_length': best_incorrect.path_length
                    }
                })

        # Sort by quality criteria including SC correct %
        paired_examples.sort(
            key=lambda x: (
                x['metrics']['sc_correct_percent'],  # Higher correct % in SC answers
                x['metrics']['sc_score'],  # Higher overall SC
                x['positive']['prm_score'],  # Higher positive score
                x['negative']['prm_score'],  # Higher negative score (more deceptive)
            ),
            reverse=True
        )
        
        return paired_examples[:max_pairs]
     
    def generate_prm_training_data(self, analyses: List[QuestionAnalysis]) -> List[Dict[str, Any]]:
        """Generate training data for Process Reward Model (PRM) from MCTS paths."""
        prm_examples = []
        original_correct_lengths = []
        original_incorrect_lengths = []
        
        for analysis in analyses:
            if analysis.sc_score < 0.6:
                continue
                
            # Process correct paths
            for path in analysis.correct_paths:
                if not path.steps:  # Skip if no steps after preprocessing
                    continue
                    
                original_correct_lengths.append(len(path.steps))
                K = len(path.steps)
                v_prev = 0
                
                for k, step in enumerate(path.steps, 1):
                    partial_steps = path.steps[:k]
                    m_k = K - k
                    r_s_k = 0
                    w_s_k = (1 - v_prev) / (m_k + 1) * (1 - 2 * r_s_k)
                    v_k = max(v_prev + w_s_k, 0)
                    
                    prm_examples.append({
                        "question": analysis.question_text,
                        "steps": partial_steps,
                        "final_step_reward": float(v_k),
                        "metadata": {
                            "is_complete": k == K,
                            "is_correct": True,
                            "path_length": K,
                            "step_number": k,
                            "raw_path_length": len(path.raw_steps)
                        }
                    })
                    v_prev = v_k
            
            # Process incorrect paths
            for path in analysis.incorrect_paths:
                if not path.steps:  # Skip if no steps after preprocessing
                    continue
                    
                original_incorrect_lengths.append(len(path.steps))
                K = len(path.steps)
                v_prev = 0
                
                for k, step in enumerate(path.steps, 1):
                    partial_steps = path.steps[:k]
                    penalize = k == K
                    m_k = K - k if not penalize else K - k + 1
                    r_s_k = 0 if not penalize else 1
                    w_s_k = (1 - v_prev) / (m_k + 1) * (1 - 2 * r_s_k)
                    v_k = max(v_prev + w_s_k, 0)
                    
                    prm_examples.append({
                        "question": analysis.question_text,
                        "steps": partial_steps,
                        "final_step_reward": float(v_k),
                        "metadata": {
                            "is_complete": k == K,
                            "is_correct": False,
                            "path_length": K,
                            "step_number": k,
                            "raw_path_length": len(path.raw_steps)
                        }
                    })
                    v_prev = v_k
                    
        # Record length statistics
        if original_correct_lengths:
            print("\nOriginal Path Length Statistics:")
            print(f"Correct paths mean length: {np.mean(original_correct_lengths):.1f} (±{np.std(original_correct_lengths):.1f})")
        if original_incorrect_lengths:
            print(f"Incorrect paths mean length: {np.mean(original_incorrect_lengths):.1f} (±{np.std(original_incorrect_lengths):.1f})")
        
        # Print complete path statistics
        complete_correct = [ex for ex in prm_examples if ex["metadata"]["is_correct"] and ex["metadata"]["is_complete"]]
        complete_incorrect = [ex for ex in prm_examples if not ex["metadata"]["is_correct"] and ex["metadata"]["is_complete"]]
        
        print("\nComplete Path Statistics:")
        print(f"Complete correct paths: {len(complete_correct)}")
        print(f"Complete incorrect paths: {len(complete_incorrect)}")
        
        if complete_correct:
            print(f"Complete correct mean length: {np.mean([ex['metadata']['path_length'] for ex in complete_correct]):.1f}")
        if complete_incorrect:
            print(f"Complete incorrect mean length: {np.mean([ex['metadata']['path_length'] for ex in complete_incorrect]):.1f}")
        
        return prm_examples

def main():
    # analyzer = MathReasoningAnalyzer('mcts_results.jsonl')
    analyzer = MathReasoningAnalyzer('mcts_results.jsonl.st0.bak')
    
    # Analyze all questions
    analyses = []
    for question in analyzer.questions:
        analysis = analyzer.analyze_question(question)
        analyses.append(analysis)
    
    # Calculate overall statistics
    total = len(analyses)
    binary_success = sum(1 for a in analyses if a.binary_success)
    avg_sc = np.mean([a.sc_score for a in analyses])
    avg_sc_correct = np.mean([a.sc_correct_percent for a in analyses])
    sc_accuracy = sum(1 for a in analyses if a.sc_most_common_correct) / total * 100
    
    # Terminal path statistics
    total_paths = [a.total_paths for a in analyses]
    correct_paths = [len(a.correct_paths) for a in analyses]
    incorrect_paths = [len(a.incorrect_paths) for a in analyses]
    
    # Path length statistics
    all_correct_lengths = [p.path_length for a in analyses for p in a.correct_paths]
    all_incorrect_lengths = [p.path_length for a in analyses for p in a.incorrect_paths]
    
    # PRM score statistics
    all_correct_scores = [p.prm_score for a in analyses for p in a.correct_paths]
    all_incorrect_scores = [p.prm_score for a in analyses for p in a.incorrect_paths]

    # Best path analysis
    best_paths_correct = 0
    total_questions = len(analyses)
    
    for question in analyzer.questions:
        # Get highest scoring path
        best_path = max(question['terminal_paths'], key=lambda x: x['score'])
        if best_path['correct']:
            best_paths_correct += 1
            
    best_path_accuracy = (best_paths_correct / total_questions) * 100
    
    print("\nBest Path Analysis:")
    print(f"Questions where highest scoring path was correct: {best_paths_correct} ({best_path_accuracy:.1f}%)")
    
    print("\nOverall Statistics:")
    print(f"Total questions analyzed: {total}")
    print(f"Questions with at least one correct path: {binary_success} ({binary_success/total*100:.1f}%)")
    print(f"Accuracy using most common answer (SC): {sc_accuracy:.1f}%")
    print(f"Average self-consistency score: {avg_sc:.3f}")
    print(f"Average % of self-consistent answers that are correct: {avg_sc_correct*100:.1f}%")
    
    print("\nTerminal Path Statistics:")
    print(f"Average total paths per question: {np.mean(total_paths):.1f} (±{np.std(total_paths):.1f})")
    print(f"Average correct paths per question: {np.mean(correct_paths):.1f} (±{np.std(correct_paths):.1f})")
    print(f"Average incorrect paths per question: {np.mean(incorrect_paths):.1f} (±{np.std(incorrect_paths):.1f})")
    
    print("\nPath Length Statistics:")
    if all_correct_lengths:
        print(f"Average correct path length: {np.mean(all_correct_lengths):.1f} (±{np.std(all_correct_lengths):.1f})")
    if all_incorrect_lengths:
        print(f"Average incorrect path length: {np.mean(all_incorrect_lengths):.1f} (±{np.std(all_incorrect_lengths):.1f})")
    
    print("\nPRM Score Statistics:")
    if all_correct_scores:
        print(f"Average correct path PRM score: {np.mean(all_correct_scores):.3f} (±{np.std(all_correct_scores):.3f})")
    if all_incorrect_scores:
        print(f"Average incorrect path PRM score: {np.mean(all_incorrect_scores):.3f} (±{np.std(all_incorrect_scores):.3f})")
    
    # Distribution of number of paths
    path_counts = Counter(total_paths)
    print("\nPath Count Distribution:")
    for count in sorted(path_counts.keys()):
        questions = path_counts[count]
        print(f"{count} paths: {questions} questions ({questions/total*100:.1f}%)")
    
    print("\nSelf-Consistency Breakdown:")
    sc_thresholds = [0.2, 0.4, 0.6, 0.8]
    for threshold in sc_thresholds:
        questions_above = sum(1 for a in analyses if a.sc_score >= threshold)
        correct_above = sum(1 for a in analyses 
                          if a.sc_score >= threshold and a.sc_most_common_correct)
        print(f"Questions with SC >= {threshold:.1f}: {questions_above} "
              f"({questions_above/total*100:.1f}%) - "
              f"Correct: {correct_above} ({correct_above/questions_above*100:.1f}% of SC)")
    
    should_generate=True
    if should_generate:
        # Generate both preference pairs and PRM training data
        paired_examples = analyzer.get_paired_examples(analyses)
        prm_training_data = analyzer.generate_prm_training_data(analyses)
        
        print(f"\nSelected {len(paired_examples)} paired examples")
        
        # Statistics on selected pairs
        if paired_examples:
            print("\nSelected Pairs Statistics:")
            pair_pos_prm = [ex['positive']['prm_score'] for ex in paired_examples]
            pair_neg_prm = [ex['negative']['prm_score'] for ex in paired_examples]
            pair_pos_len = [ex['positive']['path_length'] for ex in paired_examples]
            pair_neg_len = [ex['negative']['path_length'] for ex in paired_examples]
            pair_sc = [ex['metrics']['sc_score'] for ex in paired_examples]
            pair_sc_correct = [ex['metrics']['sc_correct_percent'] for ex in paired_examples]
            
            print("\nPaired Examples Metrics:")
            print(f"Average positive path length: {np.mean(pair_pos_len):.1f} (±{np.std(pair_pos_len):.1f})")
            print(f"Average negative path length: {np.mean(pair_neg_len):.1f} (±{np.std(pair_neg_len):.1f})")
            print(f"Average positive PRM score: {np.mean(pair_pos_prm):.3f} (±{np.std(pair_pos_prm):.3f})")
            print(f"Average negative PRM score: {np.mean(pair_neg_prm):.3f} (±{np.std(pair_neg_prm):.3f})")
            print(f"Average self-consistency: {np.mean(pair_sc):.3f} (±{np.std(pair_sc):.3f})")
            print(f"Average % correct in SC: {np.mean(pair_sc_correct)*100:.1f}% (±{np.std(pair_sc_correct)*100:.1f}%)")
        
        # Print PRM Training Data Statistics
        print("\nPRM Training Data Statistics:")
        correct_examples = [ex for ex in prm_training_data if ex["metadata"]["is_correct"]]
        incorrect_examples = [ex for ex in prm_training_data if not ex["metadata"]["is_correct"]]
        
        print(f"Total training examples: {len(prm_training_data)}")
        print(f"Correct examples: {len(correct_examples)}")
        print(f"Incorrect examples: {len(incorrect_examples)}")
        
        print("\nCorrect Examples Statistics:")
        if correct_examples:
            complete_correct = [ex for ex in correct_examples if ex["metadata"]["is_complete"]]
            print(f"Complete paths: {len(complete_correct)}")
            print(f"Average steps: {np.mean([len(ex['steps']) for ex in correct_examples]):.1f}")
            print(f"Average reward: {np.mean([ex['final_step_reward'] for ex in correct_examples]):.3f}")
        else:
            print("No correct examples found")
        
        print("\nIncorrect Examples Statistics:")
        if incorrect_examples:
            complete_incorrect = [ex for ex in incorrect_examples if ex["metadata"]["is_complete"]]
            print(f"Complete paths: {len(complete_incorrect)}")
            print(f"Average steps: {np.mean([len(ex['steps']) for ex in incorrect_examples]):.1f}")
            print(f"Average reward: {np.mean([ex['final_step_reward'] for ex in incorrect_examples]):.3f}")
        else:
            print("No incorrect examples found")
            
        # Add path length distribution
        print("\nPath Length Distribution:")
        correct_lengths = [len(ex['steps']) for ex in correct_examples]
        incorrect_lengths = [len(ex['steps']) for ex in incorrect_examples]
        
        if correct_lengths:
            correct_dist = Counter(correct_lengths)
            print("\nCorrect path lengths:")
            for length in sorted(correct_dist.keys()):
                count = correct_dist[length]
                percent = (count / len(correct_lengths)) * 100
                print(f"{length} steps: {count} examples ({percent:.1f}%)")
                
        if incorrect_lengths:
            incorrect_dist = Counter(incorrect_lengths)
            print("\nIncorrect path lengths:")
            for length in sorted(incorrect_dist.keys()):
                count = incorrect_dist[length]
                percent = (count / len(incorrect_lengths)) * 100
                print(f"{length} steps: {count} examples ({percent:.1f}%)")
                
        # Add reward distribution
        print("\nReward Distribution:")
        if correct_examples:
            correct_rewards = [ex['final_step_reward'] for ex in correct_examples]
            print(f"\nCorrect rewards: min={min(correct_rewards):.3f}, "
                f"mean={np.mean(correct_rewards):.3f}, "
                f"max={max(correct_rewards):.3f}")
                
        if incorrect_examples:
            incorrect_rewards = [ex['final_step_reward'] for ex in incorrect_examples]
            print(f"Incorrect rewards: min={min(incorrect_rewards):.3f}, "
                f"mean={np.mean(incorrect_rewards):.3f}, "
                f"max={max(incorrect_rewards):.3f}")
        
        # Save paired examples in JSONL format
        with open('paired_examples.jsonl', 'w') as f:
            for example in paired_examples:
                json.dump(example, f)
                f.write('\n')
        
        # Save PRM training data in JSONL format
        with open('prm_training.jsonl', 'w') as f:
            for example in prm_training_data:
                json.dump(example, f)
                f.write('\n')
        
        print("\nOutput files written:")
        print("- paired_examples.jsonl")
        print("- prm_training.jsonl")

if __name__ == "__main__":
    main()