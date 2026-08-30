# Running the Project

Open PowerShell and navigate to the project directory.

### 1. Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Install the requirements

```powershell
python -m pip install -r requirements.txt
```

### 3. Play the game manually

To play the game yourself:

```powershell
python main.py
```

### 4. Train the PPO agent

Run:

```powershell
python train.py
```

The training configuration is controlled through `config.py`. Depending on the selected ablation setting, different models are trained and saved.

### 5. Run headless evaluation

Run:

```powershell
python evaluate.py
```

This evaluates the trained agent without rendering the game and creates CSV result files under the `results/` folder.

### 6. Watch the trained agent play

Select the desired ablation setting in `config.py`, then run:

```powershell
python evaluate_with_head.py
```

This runs the evaluation with rendering so that the agent's behavior can be observed.

### 7. View TensorBoard charts

If TensorBoard logs do not already exist, run `train.py` first.

Then run:

```powershell
tensorboard --logdir .\logs --host 127.0.0.1 --port 0
```

PowerShell will display an address similar to:

```text
http://127.0.0.1:12986/
```

Open the displayed address in a web browser to view the TensorBoard charts.

### 8. Test the random agent

Run:

```powershell
python random_agent.py
```

This runs an untrained random agent for comparison.

---

# Project Layout

```text
project/
├── game/                                      # Core game engine
│   ├── config.py                              # Ablation and other settings
│   ├── entities.py
│   ├── environment.py
│   ├── game.py
│   ├── level.py
│   ├── player.py
│   └── __init__.py
├── logs/                                      # TensorBoard logs
├── results/                                   # Evaluation CSV results
├── evaluate.py                                # Headless evaluation
├── evaluate_with_head.py                      # Evaluation with rendering
├── main.py                                    # Manual game
├── ppo_platformer.zip                         # Successful training model
├── ppo_platformer_no_enemy_observation.zip    # Model without enemy observations
├── ppo_platformer_no_hazard_reward.zip        # Model without hazard-specific rewards
├── ppo_platformer_no_movement_reward.zip      # Model without movement rewards
├── random_agent.py                            # Untrained random agent
├── README.md
├── requirements.txt
└── train.py
```