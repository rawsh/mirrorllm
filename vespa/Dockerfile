FROM vespaengine/vespa:latest

RUN echo '#!/usr/bin/env bash' > /entry.sh
RUN echo 'exec "$@"' >> /entry.sh
RUN chmod +x /entry.sh
# ENTRYPOINT /entry.sh
ENTRYPOINT tail -f /dev/null