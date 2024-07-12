# Galaga_RL_AI

## Prequisties
### Python Packages
OpenAI Gym-Retro:
This will require a little bit more work to set-up than simply installing the package. Go to this link to learn more about setting up the enviroment properly https://retro.readthedocs.io/en/latest/getting_started.html. Keep in mind that this code is intended to take in a pixel representation of the game board.

TensorFlow 2:
You will also need the apprioate CUDA and CUDNN versions installed if you have a compatiable Nividia GPU. There is no current compatibility with Radeon GPUs.

Numpy

CV2

matplotlib

### Other
You will need the actual ROM for _Galaga: Demons of Death_. The link for getting set-up for Gym-Retro has suggestions for where you can download it from.

## Summary
This project was done for a school project in a Reinforcement Learning class at Utah State University using the shared resources at the Center for High Performance Computing hosted by University of Utah. Becuase of this, you may find that the training times differ in your particular runtime system. In theory, this should not change the results too drastically and you should see similar results to those stored in Progress_Reports/[DDQN_v3, DQN_v3]. Most documentation and results will be in the Progress_Reports folder. Overall, the project did not result in an agent that reliably learns to play _Galaga_ even though it had some promising results. There is a chance that the agents will learn to play the game properly with enough time, but the documentation and results will still hopefully help everyone improve upon my failures. I make no garentees that I will update or maintain this code, but I do hope to return to this project in the future if only as a matter of pride. For more details I recommend you read my report in the Progress_Reports folder, I also give credit where credit is due in that report.

## Running the Code
### Training
The Training of a particular agent is achieving by running main.py. The current version of both agents is version 3; this is the best version in my opinion. Before using the command python main.py in your console, make sure that line 14 is initializing the correct agent from the agents.py file. Usually all you will need to do is to add or remove a "D" from the front of the initlization statement. Note that setting debug equal at line 32 to true will slow down performance as it renders the agent training for every episode. For more information on how to initlizes both the DDQN and DQN agents look at the constructor statements in agents.py. The default number of training episodes is 500 but can be easily changed in the main.py file at line 29.

### Testing
The testing of a particular agent is done by running test_weights.py which will have the chosen agent play an episode for each set of weights saved during training at intervals of 25 episodes. Once again, the particular agent is changed  by editing the file by adding or removing a "D" to the intialization statement. This is on line 13 of test_weights.py.

### Viewing Results
During both training and testing of the agents, the code saves relevant graphs, neural network weights, and "movies" to either the DQN or DDQN folders. The "movies" are saved as .bk2 files that are actually just a record of steps taken for the particular episode. It is necessary to run galaga_AI_playback.py to render these steps for the viewer to see. The path to the particular file that the program renders must be changed each time you want to view a different episode of play. This is done at lines 4 through 5. Line 4 stores the path of the parent directory relative to the home directory. This means that you can render my previously recorded episodes and/or your own that were stored in the ./[DQN, DDQN] folders. Next, change line 5 to match the filename of the "movie" you want to watch. Normally, they follow the form "GalagaDemonsOfDeath-Nes-1Player.Level1-xxxxxx.bk2" with x representing the number of the particular episode.

## Acknowledgements
First off, I never would've have even attempted this project if it were not for my Professor and his TA who helped me learn the necessary concepts. It was unprofessional to thank them in my report, so I would like to personally thank Dr. Nicholas Flann and Christopher Brown for helping me climb onto the shoulders of giants. I would also like to thank the patient reader of this README who got this far, slogging through my rambling words.

Thank You
