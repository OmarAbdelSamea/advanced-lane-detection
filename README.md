# advanced-lane-detection
Simple Perception Stack for Self-Driving Cars Self-driving cars have piqued human interest for centuries. Leonardo Da Vinci sketched out the plans for a hypothetical self-driving cart in the late 1400s, and mechanical autopilots for airplanes emerged in the 1930s. In the 1960s an autonomous vehicle was developed as a possible moon rover for the Apollo astronauts. A true self-driving car has remained elusive until recently. Technological advancements in global positioning (GPS), digital mapping, computing power, and sensor systems have finally made it a reality In this project we are going to create a simple perception stack for self-driving cars (SDCs.) Although a typical perception stack for a self-driving car may contain different data sources from different sensors (ex.: cameras, lidar, radar, etc…), we’re only going to be focusing on video streams from cameras for simplicity. We’re mainly going to be analyzing the road ahead, detecting the lane lines, detecting other cars/agents on the road, and estimating some useful information that may help other SDCs stacks.