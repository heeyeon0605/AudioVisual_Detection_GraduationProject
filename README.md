# AudioVisual_GraduationProject

# 연구 배경 

# 관련연구

# 제안 연구의 중요성/독창성 
이전에 연구되었던 논문의 문제점은 소리가 나는 객체를 찾아야하는 연구 목적과 달리 모델이 오디오 데이터가 크게 활용되지 않고 이미지 내에서 보이는 객체를 찾는 것이다. 
구체적으로, 소리를 내는 아기 대신 수건이나 어른을 검출하거나 총, 또는 여러 악기를 검출하는 대신 악기를 연주하는 사람이나 총을 쏘는 사람 등 다른 객체를 검출하는 문제가 존재한다는 것이다. 
이에 소리 데이터를 활발히 활용하여 소리를 내는 객체만이 탐지할 수 있도록 하여 성능을 높이는 것이 본 연구의 목표이다.


## 전체 연구 내용(제안 방법 및 실험)
먼저 각 동영상의 5장의 후보 영역 토대로 자르고 resize한 이미지를 닮아가도록 한 후, 해당 loss를 오디오와 Fusion할 때 적용하였다.
Fusion을 진행할 때 이미지와 오디오를 Transformer 학습 방식을 이용하여 key, query, value값을 이미지와 오디오가 서로 닮아가도록 적절히 바꿔가며 설정하여 최적의 네트워크를 설계하도록 하여 다양한 시도를 해보았지만, 기존의 TPAVI Module 설계가 가장 높은 performance를 보였기에 기존의 Module을 그대로 사용하였다. 

기존에 존재하는 AVSBench 데이터 셋에서 하나의 5 초 길이의 동영상을 1 초 단위로 나뉘어 하나의 동영상마다 총 5 장의 이미지 데이터가 존재하며, 총 약 17000 장, 23개의 Class가 있어 레이블 작업을 함과 동시에 모델 네트워크를 설계하여 적용해 보았다. 이때, 기존 데이터에는 하나의 동영상 당 1개의 ground truth mask만 존재하였기에 추가적인 Contrastive Learning에 필요한 추가적인 Labeling을 진행하였다. 학습 데이터를 오픈소스인 labelImg 프로그램을 이용하여 Bounding Box 후보 영역 좌표 Label 값을 가져왔으며, 이후 해당 좌표에 맞게 crop 하고 원래 이미지 크기와 동일하게 모델 네트워크에 input 으로 들어갈 수 있게 resizing을 진행하였다.

이후, Resnet50 모델을 사용하여 Segmentation을 진행하였으며, Cropped Image간의 PairwiseDistance() Loss와 Label값과 Cropped Image를 입력값으로 한 Cross Entropy Loss를 더한 Total Loss를 기존 AVSModel에 추가하여 학습을 진행하였다.

추가로 오디오 Feature도 Loss와 함께 고려되기 위해 5개의 오디오를 평균낸 후 각각의 Cropped Image와 코사인 유사도를 구하여 새 Loss에 추가하였다.

그러므로 총 언급하였던 Cropped Image의 Cross Entropy Loss, PairwiseDistance Loss, 마지막으로 평균 낸 오디오와의 Loss를 포함한 3가지 Loss를 기존 AVSModel Loss에 추가시켰다.

이에 기존의 AVSModel Loss와 앞서 언급하였던 새롭게 추가한 3가지 Loss를 모두 더해 Total Loss를 함께 Backward()하여 Train하였고, 이를 기반으로 모델 Test도 진행하였다. 이때, Cross Entropy와 PairwiseDistance는 pretrained ResNet-50 모델에 Cropped and Resized Image를 input으로 넣어 학습한 결과로 받아왔다. 

train의 경우에는, AVSModel에 사용된 ResNet-50모델을 학습함과 동시에 본 연구에서 진행하는 Image Segmentation Loss를 추가하기 위하여 새로운 pretrained ResNet-50 모델 학습을 함께 진행하였다. 모델 구조를 구체적으로 설명하면 nn.Sequential을 사용하여 ResNet-50의 마지막 두 계층을 제외한 모든 계층을 가져왔으며, AvgPool2d를 사용하여 7x7 특성 맵을 평균 풀링하여 크기를 1x1로 줄인다. 이후 두 번의 선형 계층을 지나 입력 크기 2048에서 512, 그리고 최종적으로 출력 크기를 23의 특성 백터로 매핑시켜 최종 분류 결과를 생성하여 Relu 함수를 거친다. 이처럼 해당 모델 학습을 통하여 사전에 학습된 ResNet-50의 특성 추출 부분을 활용하여 입력 이미지에 대한 시각적 정보를 추출하고, 선형 계층을 통해 해당 이미지의 클래스를 분류하였다.

## 결과
기존 TPAVI만 적용한 AVSModel Train 및 Test 결과와 본 연구에서 제안한 Loss를 추가한 AVSModel의 Train 및 Test Best Miou와 F_Score를 비교한 결과, 기존 논문에 제시되었던 성능인 Best Miou를 고려한 Mf test 지표가 0.679였던 반면, 새로운 Loss를 추가한 Proposed method의 결과는 0.726으로 성능이 향상된 것을 볼 수 있었다. F_Score 또한 0.79에서 0.8449로 향상된 것을 볼 수 있었다. 

이에 


# References
[1] labelImg, https://github.com/heartexlabs/labelImg

[2] T. Baltrušaitis, C. Ahuja and L. -P. Morency, "Multimodal Machine Learning: A Survey and Taxonomy," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 41, no. 2, pp. 423-443

[3] Senocak, A., Oh, T. H., Kim, J., Yang, M. H., & Kweon, I. S. (2018). Learning to localize sound source in visual scenes. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition 

[4] Valverde, F. R., Hurtado, J. V., & Valada, A. (2021). There is more than meets the eye: Self-supervised multi-object detection and tracking with sound by distilling multimodal knowledge. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition

[5] Zhou, J., Wang, J., Zhang, J., Sun, W., Zhang, J., Birchfield, S., ... & Zhong, Y. (2022, October). Audio–Visual Segmentation. In Computer Vision–ECCV 2022: 17th European Conference

[6] Ziegler, A., & Asano, Y. M. (2022). Self-supervised learning of object parts for semantic segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition
