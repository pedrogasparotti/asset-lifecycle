import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def create_dataframe(file_path):

    # find path that contains the file
    path = Path.joinpath(Path.home(), file_path)

    raw_data = pd.read_csv(path, sep = ";")

    raw_dataframe = pd.DataFrame(raw_data)

    print(f"created dataframe from file in {path}")

    return raw_dataframe

def separate_discrete_lifecycles(df):
    """
    Split asset sequences into discrete lifecycles
    New lifecycle starts each time condition improves
    """
    all_lifecycles = []
    
    # State hierarchy for improvement detection
    state_rank = {'Ruim': 0, 'Regular': 1, 'Bom': 2}
    
    # iterate over dataframe rows
    for idx, row in df.iterrows():
        serial = row['Serial']

        # transform sequence into a list
        sequence = row['vetor_condicoes'].replace(',', ' ').split()
        sequence = [s.strip() for s in sequence if s.strip()]
        
        if not sequence:
            continue
            
        # Find improvement points (condition gets better) - edges of asset intervention
        lifecycle_starts = [0]  # Always start with first observation
        
        for i in range(1, len(sequence)):
            
            # current state vs previous state
            current_rank = state_rank.get(sequence[i], -1)
            prev_rank = state_rank.get(sequence[i-1], -1)
            
            # Improvement detected: higher rank = better condition - !END OF LIFECYCLE DETECTED!
            if current_rank > prev_rank:
                lifecycle_starts.append(i)
        
        # Add end of sequence - list containing the start and end of each lifecycle
        lifecycle_starts.append(len(sequence))
        
        # Extract each lifecycle - lifecycle dataframe with unique lifecycles from good to worse condition over time
        for lc_idx in range(len(lifecycle_starts) - 1):
            start_pos = lifecycle_starts[lc_idx]
            end_pos = lifecycle_starts[lc_idx + 1]
            
            lifecycle_sequence = sequence[start_pos:end_pos]
            
            if len(lifecycle_sequence) < 1:
                continue
                
            # Lifecycle features
            initial_state = lifecycle_sequence[0]
            final_state = lifecycle_sequence[-1]
            duration = len(lifecycle_sequence)
            
            # Did this lifecycle end in failure (Ruim)?
            failed = 1 if final_state == 'Ruim' else 0
            
            # Degradation within this lifecycle
            degraded = 0
            for j in range(len(lifecycle_sequence) - 1):
                curr_rank = state_rank.get(lifecycle_sequence[j], -1)
                next_rank = state_rank.get(lifecycle_sequence[j+1], -1)
                if next_rank < curr_rank:  # Condition got worse
                    degraded = 1
                    break
            
            all_lifecycles.append({
                'original_serial': serial,
                'lifecycle_id': f"{serial}_LC{lc_idx+1}",
                'lifecycle_sequence': ' '.join(lifecycle_sequence),
                'initial_state': initial_state,
                'final_state': final_state,
                'duration_years': duration,
                'degraded_within_lifecycle': degraded,
                'target_failed': failed,
                'is_current': 1 if lc_idx == len(lifecycle_starts) - 2 else 0  # Last lifecycle
            })
    
    lifecycle_df = pd.DataFrame(all_lifecycles)

    # Filter first lifecycles (no way to know when it started)
    lifecycle_df = lifecycle_df[~lifecycle_df['lifecycle_id'].str.contains('LC1')]

    # Lifecycles with an end
    historical_lifecycles = lifecycle_df[lifecycle_df['is_current'] == 0]

    # Lifecycles that are active
    current_lifecycles = lifecycle_df[lifecycle_df['is_current'] == 1]

    return historical_lifecycles, current_lifecycles