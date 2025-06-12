# import matplotlib.pyplot as plt
# import csv

# def load_training_data(file_path):
#     episodes = []
#     total_rewards = []
#     f2_energy = []
#     f3_duration = []
#     f4_risk = []

#     with open(file_path, mode='r') as file:
#         reader = csv.DictReader(file)
#         for row in reader:
#             episodes.append(int(row["episode"]))
#             total_rewards.append(float(row["reward"]))
#             f2_energy.append(float(row["energy_used"]))
#             f3_duration.append(float(row["mission_time"]))
#             f4_risk.append(float(row["risk_score"]))

#     return episodes, total_rewards, f2_energy, f3_duration, f4_risk


# def plot_metrics(episodes, total_rewards, f2_energy, f3_duration, f4_risk):
#     fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

#     axs[0].plot(episodes, total_rewards, label="Total Reward", color='blue')
#     axs[0].set_ylabel("Total Reward")
#     axs[0].legend()

#     axs[1].plot(episodes, f2_energy, label="Energy Efficiency (f₂)", color='green')
#     axs[1].set_ylabel("f₂ Energy Efficiency")
#     axs[1].legend()

#     axs[2].plot(episodes, f3_duration, label="Mission Duration (f₃)", color='orange')
#     axs[2].set_ylabel("f₃ Duration")
#     axs[2].legend()

#     axs[3].plot(episodes, f4_risk, label="Risk Exposure (f₄)", color='red')
#     axs[3].set_ylabel("f₄ Risk")
#     axs[3].set_xlabel("Episode")
#     axs[3].legend()

#     plt.tight_layout()
#     plt.savefig("model_output/training_metrics_plot.png")  # Optional: Save as PNG
#     plt.show()


# if __name__ == "__main__":
#     file_path = "model_output/training_log.csv"
#     episodes, total_rewards, f2_energy, f3_duration, f4_risk = load_training_data(file_path)
#     plot_metrics(episodes, total_rewards, f2_energy, f3_duration, f4_risk)
#     print("Training metrics plot generated successfully.")
import matplotlib.pyplot as plt
import csv

def load_first_episode_data(file_path):
    # Lists to store data for the first episode
    steps = []
    # These represent the *reward received at each step* for the first episode
    step_rewards = []
    # These represent the *cumulative* values at each step for the first episode
    cumulative_energy_used = []
    cumulative_mission_time = []
    cumulative_risk_score = []

    # And the calculated f-scores at each step (these are what you probably want to see)
    f2_energy_score = []
    f3_duration_score = []
    f4_risk_score = []


    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            episode_num = int(row["episode"])
            if episode_num == 0:  # Only process data for the first episode (episode 0)
                steps.append(int(row["step"]))
                step_rewards.append(float(row["reward"])) # This is the reward for *that specific step*

                # Extract cumulative values
                cumulative_energy_used.append(float(row["energy_used"]))
                cumulative_mission_time.append(float(row["mission_time"]))
                cumulative_risk_score.append(float(row["risk_score"]))

                # Extract f-scores
                f2_energy_score.append(float(row["f2_energy"]))
                f3_duration_score.append(float(row["f3_time"]))
                f4_risk_score.append(float(row["f4_risk"]))

            elif episode_num > 0: # Stop once we've passed the first episode
                break
    
    return (steps, step_rewards, cumulative_energy_used, cumulative_mission_time,
            cumulative_risk_score, f2_energy_score, f3_duration_score, f4_risk_score)


def plot_first_episode_metrics(steps, step_rewards, cumulative_energy_used, cumulative_mission_time,
                               cumulative_risk_score, f2_energy_score, f3_duration_score, f4_risk_score):

    fig, axs = plt.subplots(5, 1, figsize=(12, 16), sharex=True) # Increased rows for step reward

    # Plot 1: Reward per Step
    axs[0].plot(steps, step_rewards, label="Reward per Step", color='purple')
    axs[0].set_ylabel("Reward")
    axs[0].set_title("Metrics for First Episode (Episode 0)")
    axs[0].legend()
    axs[0].grid(True)


    # Plot 2: F2 Energy Efficiency Score (This is what you probably want from f2_energy)
    axs[1].plot(steps, f2_energy_score, label="f₂ Energy Efficiency Score", color='green')
    axs[1].set_ylabel("f₂ Score")
    axs[1].legend()
    axs[1].grid(True)

    # Plot 3: F3 Mission Duration Score (This is what you probably want from f3_duration)
    axs[2].plot(steps, f3_duration_score, label="f₃ Mission Duration Score", color='orange')
    axs[2].set_ylabel("f₃ Score")
    axs[2].legend()
    axs[2].grid(True)

    # Plot 4: F4 Risk Exposure Score (This is what you probably want from f4_risk)
    axs[3].plot(steps, f4_risk_score, label="f₄ Risk Exposure Score", color='red')
    axs[3].set_ylabel("f₄ Score")
    axs[3].legend()
    axs[3].grid(True)

    # Plot 5: Cumulative Resource Usage (for context)
    axs[4].plot(steps, cumulative_energy_used, label="Cumulative Energy Used", color='blue', linestyle='--')
    axs[4].plot(steps, cumulative_mission_time, label="Cumulative Mission Time", color='cyan', linestyle=':')
    axs[4].plot(steps, cumulative_risk_score, label="Cumulative Risk Score", color='magenta', linestyle='-.')
    axs[4].set_ylabel("Cumulative Values")
    axs[4].set_xlabel("Step within Episode")
    axs[4].legend()
    axs[4].grid(True)


    plt.tight_layout()
    plt.savefig("model_output/first_episode_metrics_plot2.png")
    plt.show()


if __name__ == "__main__":
    file_path = "model_output/training_log.csv"
    (steps, step_rewards, cumulative_energy_used, cumulative_mission_time,
     cumulative_risk_score, f2_energy_score, f3_duration_score, f4_risk_score) = load_first_episode_data(file_path)

    if steps: # Check if data for the first episode was loaded
        plot_first_episode_metrics(steps, step_rewards, cumulative_energy_used, cumulative_mission_time,
                                   cumulative_risk_score, f2_energy_score, f3_duration_score, f4_risk_score)
        print("Metrics plot for the first episode generated successfully.")
    else:
        print(f"No data found for Episode 0 in {file_path}. Please ensure the log file exists and contains data for episode 0.")