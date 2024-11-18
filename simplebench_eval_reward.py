import aiohttp
import asyncio
import pandas as pd
import json
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm

MODAL_ENDPOINT = "https://rawsh--reward-api-model-score-dev.modal.run"
MAX_CONCURRENT = 8

async def get_score(sem, session, messages, question_id, option_num, answer, is_correct):
    async with sem:
        try:
            async with session.post(
                MODAL_ENDPOINT, 
                json={"messages": messages},
                headers={"Content-Type": "application/json", "Accept": "application/json"}
            ) as response:
                if response.status != 200:
                    print(f"Error {response.status}: {await response.text()}")
                    score = 0
                else:
                    result = await response.json()
                    score = result.get('score', 0)
                
                return {
                    'question_id': question_id,
                    'option_num': option_num,
                    'answer': answer,
                    'score': float(score),
                    'is_correct': is_correct
                }
        except Exception as e:
            print(f"Exception: {str(e)}")
            return {
                'question_id': question_id,
                'option_num': option_num,
                'answer': answer,
                'score': 0,
                'is_correct': is_correct
            }

async def evaluate_all(session, df):
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    
    print("Preparing requests...")
    all_requests = []
    
    for _, row in df.iterrows():
        question = row['question_func']
        correct_answer = row['correct_answer']
        
        # First evaluate the correct answer as option0
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": correct_answer}
        ]
        
        all_requests.append(get_score(
            sem,
            session,
            messages,
            row['id'],
            'option0',
            correct_answer,
            True
        ))

        option_keys = ['option1', 'option2', 'option3', 'option4', 'option5']
        possible_answers = "\n".join([row[option_key] for option_key in option_keys])
        
        # Then evaluate all other options
        for i, option in enumerate(option_keys, 1):
            messages = [
                {"role": "user", "content": f"Return only the answer. {question}"},
                {"role": "assistant", "content": f"{row[option]}"}
            ]
            
            all_requests.append(get_score(
                sem,
                session,
                messages,
                row['id'],
                f'option{i}',
                row[option],
                row[option] == correct_answer
            ))
    
    print(f"\nEvaluating {len(all_requests)} options (max {MAX_CONCURRENT} concurrent)...")
    all_scores = await tqdm_asyncio.gather(*all_requests)
    
    results_by_question = {}
    for score in all_scores:
        qid = score['question_id']
        if qid not in results_by_question:
            results_by_question[qid] = []
        results_by_question[qid].append(score)
    
    all_results = []
    
    print("\nProcessing results...")
    for qid in tqdm(results_by_question.keys()):
        scores = results_by_question[qid]
        scores.sort(key=lambda x: x['score'], reverse=True)
        
        question_row = df[df['id'] == qid].iloc[0]
        
        print(f"\nEvaluating Question {qid}:")
        print(f"Question: {question_row['question_func']}")
        print(f"Correct Answer: {question_row['correct_answer']}")
        print("\nScores (sorted by highest first):")
        
        for score_data in scores:
            print(f"Option: {score_data['option_num']}")
            print(f"Answer: {score_data['answer']}")
            print(f"Score: {score_data['score']}")
            print(f"Is Correct: {score_data['is_correct']}")
            print("---")
        
        correct_scores = [s for s in scores if s['is_correct']]
        if correct_scores:
            correct_score = correct_scores[0]
            correct_rank = scores.index(correct_score) + 1
            
            result = {
                'question_id': qid,
                'correct_rank': correct_rank,
                'total_options': len(scores),
                'score_diff': scores[0]['score'] - correct_score['score'],
                'correct_score': correct_score['score'],
                'best_score': scores[0]['score']
            }
            
            print(f"\nCorrect answer rank: {correct_rank} out of {len(scores)}")
            print(f"Correct answer score: {correct_score['score']:.4f}")
            print(f"Best score: {scores[0]['score']:.4f}")
            print(f"Score difference: {result['score_diff']:.4f}")
            
            all_results.append(result)
        
        print("\n" + "="*50)
    
    return all_results

async def main():
    try:
        df = pd.read_csv('simple_bench_public.csv')
        print(f"Loaded {len(df)} questions from CSV")
        
        async with aiohttp.ClientSession() as session:
            results = await evaluate_all(session, df)
            
            if results:
                print("\nSummary Statistics:")
                df_results = pd.DataFrame(results)
                print(f"Average rank of correct answer: {df_results['correct_rank'].mean():.2f}")
                print(f"Times correct answer ranked first: {len(df_results[df_results['correct_rank'] == 1])}/{len(df_results)}")
                print(f"Average score difference from best: {df_results['score_diff'].mean():.4f}")
                print(f"Average correct answer score: {df_results['correct_score'].mean():.4f}")
                print(f"Average best score: {df_results['best_score'].mean():.4f}")
    
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())