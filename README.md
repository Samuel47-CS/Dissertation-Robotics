# Dissertation Code and Robotics

# Diss TODO

Title: Human Robot Interfaces for Interactive Task Learning 
Subtitle: Teaching a robot to exhibit style throughout a task

1. Write out bullets about every single thing I have done for presentation / report
2. Train robot arms on folding (Neural Network)
    - MLP cluster scripts! https://github.com/assistive-autonomy/slurm-cluster-scripts/ - Slides in diss folder
        - Style 2 and 3 still to be trained
        - sbatch --job-name=Style1 -o /home/$USER/slogs/sl_l_%A.out -e /home/$USER/slogs/sl_l_%A.out -N 1 -n 1 --nodelist="landonia25" --gres=gpu:1 --mem-per-cpu=6000 --partition=Teaching -t 1-00:00:00  --cpus-per-task=10 Dissertation-Robotics/cluster_training/slurm_arrayjob_style_1.sh
        - sbatch --job-name=Style2 -o /home/$USER/slogs/sl_l_%A.out -e /home/$USER/slogs/sl_l_%A.out -N 1 -n 1 --nodelist="landonia25" --gres=gpu:1 --mem-per-cpu=6000 --partition=Teaching -t 1-00:00:00  --cpus-per-task=10 Dissertation-Robotics/cluster_training/slurm_arrayjob_style_2.sh
        - sbatch --job-name=Style3 -o /home/$USER/slogs/sl_l_%A.out -e /home/$USER/slogs/sl_l_%A.out -N 1 -n 1 --nodelist="landonia25" --gres=gpu:1 --mem-per-cpu=6000 --partition=Teaching -t 1-00:00:00  --cpus-per-task=10 Dissertation-Robotics/cluster_training/slurm_arrayjob_style_3.sh
        - sbatch Dissertation-Robotics/cluster_training/slurm_arrayjob_style_2.sh
        - sbatch Dissertation-Robotics/cluster_training/slurm_arrayjob_style_3.sh
    - Saving files from mlp cluster to local device. scp from mlp cluster to uni account, then uni account to local
        - scp -r s2210183@mlp:/home/s2210183/outputs ~/Lerobot/
        - scp -r s2210183@student.ssh.inf.ed.ac.uk:~/Lerobot/ ~/Documents/University/Year\ 4/Dissertation/Training/
3. Model inference:
    - Fix async inference pipeline
    - Video examples. Reason the difference in motion (hopefully there is a difference in motion)
4. Write up
    - Build narrative for all choices. E.g., i did this -> there was this problem and this was my solution / I had a couple options for an approach for this but decided on this one because this


- Success Criteria
    - Prompting the policy with the task + some style indicator causes the robot to:
        1. Fold the cloth successfully (imperfect folds are okay)
        2. Fold the cloth in a manner corresponding with the prompted style indicator


## What I have done 
- Hardware 
    - Assign motor id's to the new SO101 arms
        - Required dismantling and reassembling both sets of arms, as the motors had been put in the wrong places previously (5V motors vs 12V motors)
    - Spent lots of time ensuring that arms could be calibrated properly
    - Incorporated cameras into setup (troubleshooting required was number of connections to the same port)
    - Number a total of 7 ports for the total setup including all motors and cameras. Had to upgrade to a PC (from laptop) and transfer all code over. 
        - Still had dataflow issues with multiple camera feeds going through same port extender
    - Pieced together bi-SO101 code (including calibrate, teleoperate, record) so that both arms could be used in tandem from a single command / script
    - When everything was working together, had to reduce dataflow fps and not display data during recording sessions to avoid jitter due to slow data transfer causing overcorrection
- Research
    - Background chapter research
        - Found that previous work has not had good metrics for measuring success of robot movement with a notion of style
    - Developed initial 3D representation of style (Speed, Force, Exaggeration)
- Data collection 
    - Collected 230 episodes of data
        - Attempted to make data as diverse as possible in ways of folding the cloth. Did this by getting a few different people to record examples of cloth folding (do i mention this?)
        - Kept somewhat consistent by re-recording instances where the cloth was dropped early 
        - Episodes:
            - 30 - Adrian
            - 30 - El
            - 30 - Freddie
            - 20 - Josh
            - 90 - Sam
            - 30 - Sid
- Software / Research / Methodology / idontevenknowanymore
    - Started recording data using my initial style representation as labels (Speed = mid, Force = mid, Exaggeration = high)
        - Didn't work due to consistency issues when recording - what does it mean to fold with the specified metrics? Couldn't interpret properly / even if had, i would have to consistently act as described which is even more tricky
    - Developed idea to extract difference of styles via extracted motor control sequence data 
        - Required extracting motor data from the hub in .parquet datatype - wrote script to access, clip and convert
        - Required merging datasets - discovered how to do this with phosphobot
        - Required converting the dataset into v2.1 from v3.0 - discovered how to do this via Lerobot github
        - Required changing lots of task data simultaneously (in info.json, task.jsonl, episodes.jsonl, episode_stats.jsonl) - wrote a script to edit all of these accurately and simultaneously
        - Required ensuring that the policy trainer script does use multiple task descriptions, not just one - going to do that
    - Data analysis
        - Looked at PCA and UMAP dimensionality reduction. Both averaged or abstracted over the duration of the episodes in some way, making results useless as episode duration is a factor that effects the style of a task.
        - Split each episode into three phases: pre_grasp, grasp and post_grasp. 
            - Hand tuned parameters for this to ensure the grasp phase was extracted for all episodes, as all episodes start and end with the graspers closed (as if lifting the cloth). This caused some episodes to mark the grasp phase to start and end within the first few timesteps of the episode
        - Filtering (reduce variance)
        - Used DTW as a unit of measurement of distance between individual joint paths. Used this to develop a DTW distance matrix measuring the distance between each episode (normalised and summed over all joints)
        - Used Agglomerative Clustering to label clusters based on DTW distance matrix
        - Parameter tuning for phase and joint weights - Optuna (reduce variance)
            - Retuned number of clusters using Optuna during weight tuning to ensure weights are tuned to the optimal number of clusters - used silhouette score (scikit learn) as a metric for clustering
            - Left and right arms have same weights for corresponding joints
    - Model training on mlp cluster for no style
        - Wrote a slurm script to manage policy training
            - slurm script created (and managed) individual virtual environment for each training
            - Initial training scripts kept timing out, so learned how to assign jobs to a specific GPU machine "landonia11"
    - Model training on mlp cluster for different styles
        - Found that v2.1 datasets cannot be used for training data. Labelling episodes will not work
        - Instead, three models will be trained on subsets of the trimmed dataset (trimmed of outlier episodes that cause single-episode clusters)
        - Can I then merge these models to a 'super-model' ?
    - Model evaluation:
        - Updating and fixing async-inference pipeline to use finetuned smolvla model 
- Thought for future work:
    - Subsets of episodes for training data do not need to be exclusive per style. As in an episode could be able to belong to multiple styles. Then, how do you find 