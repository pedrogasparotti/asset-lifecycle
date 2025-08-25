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
                'target_failed': failed
            })
    
    lifecycle_df = pd.DataFrame(all_lifecycles)
    
    print(f"=== DISCRETE LIFECYCLE ANALYSIS ===")
    print(f"Original assets: {df['Serial'].nunique()}")  
    print(f"Total discrete lifecycles: {len(lifecycle_df)}")
    print(f"Avg lifecycles per asset: {len(lifecycle_df) / df['Serial'].nunique():.2f}")
    print(f"Lifecycle duration distribution:")
    print(lifecycle_df['duration_years'].value_counts().sort_index())
    print(f"\nFailure rate by lifecycle:")
    print(lifecycle_df['target_failed'].value_counts())
    print(f"Overall failure rate: {lifecycle_df['target_failed'].mean()*100:.1f}%")
    
    print(lifecycle_df[lifecycle_df['target_failed'] == 1])

    return lifecycle_df

def prepare_duration_features(lifecycle_df):
    """
    Transform lifecycles into duration prediction features
    Predict years remaining until Ruim state
    """
    duration_samples = []
    
    for idx, row in lifecycle_df.iterrows():
        sequence = row['lifecycle_sequence'].split()
        
        # Create training samples from each point in the lifecycle
        for time_point in range(len(sequence)):
            current_state = sequence[time_point]
            years_elapsed = time_point + 1
            years_remaining = len(sequence) - time_point
            
            # Skip if already at Ruim (no more degradation possible)
            if current_state == 'Ruim':
                continue
                
            # Historical context features
            states_so_far = sequence[:time_point+1]
            bom_count = states_so_far.count('Bom')
            regular_count = states_so_far.count('Regular')
            
            # Degradation velocity
            changes = 0
            for i in range(len(states_so_far)-1):
                if states_so_far[i] != states_so_far[i+1]:
                    changes += 1
            
            degradation_rate = changes / years_elapsed if years_elapsed > 0 else 0
            
            # Will reach Ruim eventually in this lifecycle?
            will_reach_ruim = 1 if 'Ruim' in sequence[time_point:] else 0
            
            # Years until Ruim (target)
            years_to_ruim = None
            for future_idx in range(time_point, len(sequence)):
                if sequence[future_idx] == 'Ruim':
                    years_to_ruim = future_idx - time_point
                    break
            
            # If never reaches Ruim, use remaining lifecycle length
            if years_to_ruim is None:
                years_to_ruim = years_remaining
                
            duration_samples.append({
                'lifecycle_id': row['lifecycle_id'],
                'current_year': years_elapsed,
                'current_state': current_state,
                'years_elapsed': years_elapsed,
                'bom_ratio': bom_count / years_elapsed,
                'regular_ratio': regular_count / years_elapsed,
                'degradation_rate': degradation_rate,
                'will_reach_ruim': will_reach_ruim,
                'years_to_ruim': years_to_ruim
            })
    
    duration_df = pd.DataFrame(duration_samples)
    
    print(f"=== DURATION PREDICTION SETUP ===")
    print(f"Training samples created: {len(duration_df)}")
    print(f"Samples that reach Ruim: {duration_df['will_reach_ruim'].sum()}")
    print(f"Average years to Ruim: {duration_df[duration_df['will_reach_ruim']==1]['years_to_ruim'].mean():.1f}")
    print(f"Years to Ruim distribution:")
    print(duration_df[duration_df['will_reach_ruim']==1]['years_to_ruim'].value_counts().sort_index())
    
    return duration_df