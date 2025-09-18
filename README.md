# Autonomous-Maze-Solving-Robot-
This project features an autonomous robot designed to solve mazes by reading signs on the walls and navigating to the end goal. The robot leverages machine vision algorithms, machine learning for sign recognition, and advanced path planning and search algorithms to efficiently navigate complex environments.

####Georgia Institute of Technology: Team 21: Sagar Desai and Eemil Harkonen 



### Testing and Design
The machine vision algorithm was to be able to recognize the directionl signs and accurately label them according to their designed motion. There were two left signs (curved arrow and steaight arrow), two right signs (curved arrow and straigth arrow) and U-Turn (Denoted as a Stop Sign and U-Turn Sign) and an end goal of a Star. The objective was to navigate the maze, scan each sign using the onboard camera and classify it, and produce the desired motion and direction. Using K-Nearest-Neighbors Algorithm (KNN), we were able to classsify each signed with ~90% accuracy. The machine vision pipeline had to be lightweight due to the hardware limitations of the raspberry pi on board and directing the processing on the computer served IoT challenges such as increased traffic from the others.


<img width="1840" height="1121" alt="Screenshot 2025-09-18 at 3 15 40 PM" src="https://github.com/user-attachments/assets/6ad32580-4bee-446c-baed-73cf36db5188" />




## Demo


[![Watch the demo](https://img.youtube.com/vi/VIDEO_ID/0.jpg)](https://youtu.be/6Tk9iepP2Z4)



Led the development of the machine vision subsystem responsible for detecting and interpreting signs on maze walls. Designed and implemented computer vision algorithms using OpenCV for robust image processing under variable lighting conditions. Developed machine learning models to accurately classify and read different sign types, enabling the robot to make context-aware navigation decisions. Integrated real-time video input with the control system to provide continuous environmental feedback. Collaborated on optimizing the vision pipeline for low-latency performance essential to autonomous operation. This work directly supported the robot’s path planning by supplying reliable environmental cues for dynamic route adjustments. Eemil led the development of the motion control and navigation system, implementing dead reckoning techniques to enable precise robot localization within the maze. He designed algorithms to estimate position based on wheel encoder data and inertial measurements, ensuring accurate movement tracking despite environmental uncertainties. His work integrated closely with the machine vision system to adjust paths dynamically and reach the end goal efficiently. 
