import os 
import csv
import torch


import WayTu_RAI.model_utils as mutils
from WayTu_RAI.models.WayTU_Model import WayTuModel
from WayTu_RAI.GenerateEnvironment import GenerateEnvironment, PlacementError


def test_environment(cfg):
    model_path = cfg["model_path"]
    
    # Load the best model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = WayTuModel(cfg, device).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(device)

    
    for i in range(cfg["num-trials"]):
        environment = GenerateEnvironment(cfg)
        environment.generate_environment()

        env_pcl = environment.point_clouds_labels

        env_pc = env_pcl[:, :3]
        env_label = env_pcl[:, 3]

        # I need to create the graph before giving the model
        environment_graph = mutils.getEnvironmentGraph(env_pc, cfg["radius"])
        # The dimension of environment graph might need an update
        with torch.no_grad():
            predictions = model(environment_graph.to(device))
        
        environment.set_waypoints(predictions)
        score = mutils.ManipulationWithKOMO(environment.C, selected_tool, environment.env)

    

# Collect samples for training dataset
def collect_samples(cfg):
    save_path = cfg['dataset-save-path']
    num_trials = cfg['num-trials']

    # Created main folder for samples and get number of samples 
    if not os.path.exists(os.path.join("./", save_path)):
        os.mkdir(os.path.join("./", save_path))

        # Create csv
        columns = ["path", "task", "selected_tool", "tool-waypoint", "initial-waypoint", "goal-waypoint", "score", "grasp_score", "task_score"]
        
        csv_name = 'dataset_info.csv'
        csv_path = os.path.join(save_path, csv_name)
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns)
            writer.writeheader()
        

    total_count, success_count = 0, 0

    for trial in range(num_trials):
        total_count += 1
        try: 
            current_sample_count = len(os.listdir(save_path)) // 2

            environment = GenerateEnvironment(cfg)
            environment.generate_environment()

            env_pcl = environment.point_clouds_labels

            # Extract the point cloud and labels
            env_pc = env_pcl[:, :3]
            env_label = env_pcl[:, 3]

            # Select a random tool
            # selected_tool, tool_label = environment.select_random_tool()
            # Collect samples for a tool 
            selected_tool, tool_label = "ball", 7
            print(f"Selected tool: {selected_tool}, label: {tool_label}")
            tool_mask = env_label == tool_label 
            tool_pc = env_pc[tool_mask]

            # Get platform points
            environment_mask = env_label < 3
            environment_pc = env_pc[environment_mask] 

            other_tools = [item for item in environment.tool_objs if item != selected_tool]

            # Try to find waypoints
            waypoints = environment.set_waypoints(
                pcl_dict={"tool_pc": tool_pc, "environment_pc": environment_pc},
                selected_tool=selected_tool,
                other_tools=other_tools
            )

            # Save environment and run KOMO
            environment.save_environment(current_sample_count)
            score = mutils.ManipulationWithKOMO(environment.C, selected_tool, environment.env)

            # Save full sample
            environment.saveSample(current_sample_count, env_pcl, selected_tool, waypoints, score)
            success_count += 1
            print(f"[✓] Sample {current_sample_count} collected successfully")
        except PlacementError as e:
            print(f"[!] Skipping sample due to error: {e}")
            continue

    print(f"\nDone! Success: {success_count}/{total_count} samples.")