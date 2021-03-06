### “Automatic Detection of Face Touch and Self-adaptors in Infants”
This repository provides a framework for extracting, processing and classifying face touches in infants.
A demo explaining all the main characteristics of the code can be seen in the following file:
["Demo_Predicting_Face_Touches.ipynb"](https://github.com/bt403/openpose-research-keras/blob/main/Demo_Predicting_Face_Touches.ipynb)

## Citation
This code has used existing implementations of OpenPose for infants, 3D-FAN and MediaPipe to support in the detection and classification of the face touches.

	@article{Chambers2019, 
		author = "Claire Chambers and Nidhi Seethapathi and Rachit Saluja and Michelle Johnson and Konrad Paul Kording", 
		title = "{Computer vision to automatically assess infant neuromotor risk}", 
		year = "2019", 
		month = "8", 
		url = "https://figshare.com/articles/dataset/Video-analysis_based_automated_infant_movement_assessment_Infant_pose_data_meta_data_and_video_URLs/8161430", 
		doi = "10.6084/m9.figshare.8161430.v5" 
	} 

	@InProceedings{cao2017realtime,
		title = {Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields},
		author = {Zhe Cao and Tomas Simon and Shih-En Wei and Yaser Sheikh},
		booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
		year = {2017}
	}

	@inproceedings{bulat2017far,
		title={How far are we from solving the 2D \& 3D Face Alignment problem? (and a dataset of 230,000 3D facial landmarks)},
		author={Bulat, Adrian and Tzimiropoulos, Georgios},
		booktitle={International Conference on Computer Vision},
		year={2017}
	}
	
	@inproceedings{48292,
		title	= {MediaPipe: A Framework for Perceiving and Processing Reality},
		author	= {Camillo Lugaresi and Jiuqiang Tang and Hadon Nash and Chris McClanahan and Esha Uboweja and Michael Hays and Fan Zhang and Chuo-Ling Chang and Ming Yong and Juhyun Lee and Wan-Teh Chang and Wei Hua and Manfred Georg and Matthias Grundmann},
		year	= {2019},
		URL	= {https://mixedreality.cs.cornell.edu/s/NewTitle_May1_MediaPipe_CVPR_CV4ARVR_Workshop_2019.pdf},
		booktitle	= {Third Workshop on Computer Vision for AR/VR at IEEE Computer Vision and Pattern Recognition (CVPR) 2019}
	}


