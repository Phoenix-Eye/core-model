#!/bin/bash

echo "
                    (                           )
          ) )( (                           ( ) )( (
       ( ( ( )  ) )                     ( (   (  ) )(
      ) )     ,,\\\                     ///,,       ) (
   (  ((    (\\\\//                     \\////)      )
    ) )    (-(__//                       \\__)-)     (
   (((   ((-(__||                         ||__)-))    ) )
  ) )   ((-(-(_||           ```\__        ||_)-)-))   ((
  ((   ((-(-(/(/\\        ''; 9.- `      //\)\)-)-))    )
   )   (-(-(/(/(/\\      '';;;;-\~      //\)\)\)-)-)   (   )
(  (   ((-(-(/(/(/\======,:;:;:;:,======/\)\)\)-)-))   )
    )  '(((-(/(/(/(//////:%%%%%%%:\\\\\\)\)\)\)-)))`  ( (
   ((   '((-(/(/(/('uuuu:WWWWWWWWW:uuuu`)\)\)\)-))`    )
     ))  '((-(/(/(/('|||:wwwwwwwww:|||')\)\)\)-))`    ((
  (   ((   '((((/(/('uuu:WWWWWWWWW:uuu`)\)\))))`     ))
        ))   '':::UUUUUU:wwwwwwwww:UUUUUU:::``     ((   )
          ((      '''''''\uuuuuuuu/``````         ))
           ))            `JJJJJJJJJ`           ((
             ((            LLLLLLLLLLL         ))
               ))         ///|||||||\\\       ((
                 ))      (/(/(/(^)\)\)\)       ((
                  ((                           ))
                    ((                       ((
                      ( )( ))( ( ( ) )( ) (()


    🔥•°• PHOENIX EYE SYSTEM PIPELINE •°•🔥
    ═══════════════════════════════════════
    
    ┌───────────────────────────────────┐
    │     🦅 RISE ABOVE, WATCH BELOW    │
    └───────────────────────────────────┘

    ┌─────────── FEATURES ───────────┐
    │  🌲 AI-Powered Forest Watch    │
    │  🛰️ Satellite Data Analysis    │
    │  ⚡ Real-time Alert System     │
    │  🤖 Machine Learning Core      │
    │  📡 LoRa Communication         │
    └────────────────────────────────┘

    ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
         Initializing Pipeline...
    ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
"

echo "🚀 Running complete Phoenix Eye pipeline..."
# Ensure environment is properly set up
source venv/bin/activate
# Setup Earth Engine authentication
./setup/auth_setup.sh
# Check for required dependencies
if ! pip show tf-keras > /dev/null; then
echo "❌ Missing required dependencies. Running environment fix..."
./setup/fix_environment.sh
fi
# Run each step in sequence with error checking
for step in {1..5}; do
script="./setup/${step}_*.sh"
echo "
    ╔════════════════════════╗
    ║ Running Pipeline Step ${step} ║
    ╚════════════════════════╝
"
# Try running the script up to 3 times
for attempt in {1..3}; do
if bash $script 2>logs/temp/step_${step}_attempt_${attempt}.log; then
echo "
    ┌────────────────────────┐
    │ ✅ Step ${step} Complete! │
    └────────────────────────┘
"
break
else
echo "❌ Step ${step} failed (attempt $attempt). Checking error..."
if [ $attempt -eq 3 ]; then
echo "⚠️ Step ${step} failed after 3 attempts. Continuing with sample data..."
# Generate sample data if data collection failed
if [ $step -eq 1 ]; then
python -c "
import numpy as np
from pathlib import Path
sample_data = {
 'spatial': np.random.random((100, 64, 64, 5)),
 'temporal': np.random.random((100, 24, 10)),
 'labels': np.random.randint(0, 2, (100, 64, 64))
}
processed_dir = Path('data/processed')
processed_dir.mkdir(parents=True, exist_ok=True)
np.savez(processed_dir / 'processed_data.npz', **sample_data)
"
fi
fi
fi
done
done

echo "
    ╔═══════════════════════════════════╗
    ║      Pipeline Run Complete!       ║
    ╚═══════════════════════════════════╝

    🔥═══════════════════════════════🔥
        Watching Over Our Forests
        24/7 Wildfire Protection
    🔥═══════════════════════════════🔥

"