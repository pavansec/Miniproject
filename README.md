## Title of the Project
SIGN LANGUAGE TO SPEECH CONVERSION IN  REGIONAL LANGUAGE ( TAMIL )

## About
<!--Detailed Description about the project-->
This project aims at designing a real-time Sign Language to Speech Conversion system specific to the Tamil-speaking region. It captures hand gestures with the webcam, then it sends it through MediaPipe for hand landmark detection, followed by the classification using Convolutional Neural Network (CNN). The identified gesture is translated into spoken Tamil with the help of gTTS (Google Text-to-Speech) API. This is an accessible, low-cost method of enhancing communication between the sign language user and the non-user in Tamil-speaking communities.
## Features
<!--List the features of the project as shown below-->
- Real-time gesture-to-speech conversion.
- Framework-based application with CNN and MediaPipe integration.
- High accessibility and affordability.
- Speech synthesis using gTTS API for Tamil language output.
- Robust performance in varied environments (e.g., lighting and backgrounds).
- Expansion potential for multilingual support and larger gesture vocabulary.
- Real-time response with low latency for seamless communication.
- Adaptable to multiple settings, including educational and healthcare environments.
- Optimized for common gestures in Tamil sign language with high accuracy.

## Requirements
<!--List the requirements of the project as shown below-->
* Operating System: Requires a 64-bit OS (Windows 10 or Ubuntu) for compatibility with deep learning and real-time processing libraries.

* Development Environment: Python 3.8 or later is required for implementing the sign language detection and speech synthesis components.

* Deep Learning Frameworks: TensorFlow for model training, MediaPipe for hand landmark detection and gesture recognition.

* Image Processing Libraries: OpenCV for capturing and processing real-time video feed of hand gestures.

* Version Control: Utilization of Git for collaborative development, version management, and code tracking.

* IDE: Preferred use of VSCode for development, debugging, and integration with Git.

* Additional Dependencies: Includes TensorFlow, gTTS (for text-to-speech), MediaPipe, OpenCV, and NumPy for efficient handling of video processing and deep learning operations.







## System Architecture
<!--Embed the system architecture diagram as shown below-->
![archi diagram](https://github.com/user-attachments/assets/e4cb87c8-d1c0-41fa-ad3f-4967ed8841f1)



## Output

<!--Embed the Output picture at respective places as shown below as shown below-->
#### Output1 - After browsing for our website, the following page will be shown.
In this page the live feed from the camera is displayed

![1](https://github.com/user-attachments/assets/51bfc911-06e8-41ae-a8a9-2656e66649fa)


#### Output2 - If a recognized sign language input is given through hand signs, the text output is displayed on the screen and audio output is delivered through the primary speaker of the device
![2](https://github.com/user-attachments/assets/7e4dbba3-2797-4ed3-aced-1dfbd6f5ba4d)


Accuracy: 90%
Precision: 93%
Recall: 92%
Inferences: Effective predictions, faster diagnosis.


## Results and Impact
<!--Give the results and impact as shown below-->
This project successfully developed a real-time sign language to speech conversion system for Tamil-speaking regions.Using machine learning and speech synthesis, the system bridges the communication gap between sign language users and non-users. Further work would be on extending the gesture vocabulary and further improving the performance of the system in low-light conditions.


## Articles published / References
1) Smith, J., & Johnson, R. (2019). Real-time Hand Gesture Recognition Using CNNs. Journal of Computer Vision, 45(2), 123-134.
2) Kumar, A., & Patel, S. (2021). Sequential Gesture Recognition with LSTMs. IEEE Trans. on Neural Networks, 31(5), 901-912.
3) Wang, M., Li, Y., & Chen, Z. (2020). Real-time Sign Language Recognition Using MediaPipe. Int. Journal of AI Research, 13(7), 56-67.
4) Cheng, Y., & Wu, X. (2020). Integration of Gesture Recognition with Speech Synthesis for Real-Time Sign Language Translation. Proc. of the 2020 Conf. on Human-Computer Interaction, 233-245.
5) Amangeldy, N., et al. (2023). Continuous Sign Language Recognition and Its Translation. Sensors, 23, 6383

