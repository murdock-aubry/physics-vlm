import pandas as pd
import re
# Load CSV
df = pd.read_csv('gravity.csv')

# Create an empty list to store the extracted data
video_data = []

# Iterate through the rows of the dataframe
for _, row in df.iterrows():
    # Extract vid1 and vid2 related information
    video_data.append({
        'video_path': row['vid1_path'],
        'start_frame': row['vid1_end_priming_2']+1,
    })
    video_data.append({
        'video_path': row['vid2_path'],
        'start_frame': row['vid2_end_priming_2']+1,
    })

# Convert the list of video data to a dataframe
video_df = pd.DataFrame(video_data)
video_df['video_path'] = video_df['video_path'].str.replace(r'^(continuity/|gravity/|solidity/)', '', regex=True)
#video_df.to_csv('gravity_start_frame.csv', index=False)
print(video_df)

loaded_df = pd.read_csv('gravity_start_frame.csv',index_col='video_path')

# Convert the DataFrame to a dictionary
video_dict = loaded_df.to_dict(orient='index')
# Show the dictionary
print(video_dict)
video_title = "right__gravity__bigtealcup__redball__uv.mp4.mp4"
start_frame = video_dict[video_title]['start_frame']
print(start_frame)