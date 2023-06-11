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
이후, Resnet50 모델을 사용하여 Segmentation을 진행하였으며, Cropped Image간의 PairwiseDistance() Loss와 Label값과 Cropped Image를 입력값으로 한 Cross Entropy Loss를 더한 Total Loss를 기존 AVSModel에 추가하여 학습을 진행하였다.
이후, 오디오 Feature도 Loss와 함께 고려되기 위해 5개의 오디오를 평균낸 후 각각의 Cropped Image와 코사인 유사도를 구하여 새 Loss에 추가하였습니다.

그러므로 총 Cropped Image간의 두 Loss와 평균 낸 오디오와의 Loss를 포함한 3가지 Loss를 기존 AVSModel Loss에 추가시켰습니다.



# 제안 방법


# References
[1] labelImg, https://github.com/heartexlabs/labelImg

[2] T. Baltrušaitis, C. Ahuja and L. -P. Morency, "Multimodal Machine Learning: A Survey and Taxonomy," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 41, no. 2, pp. 423-443

[3] Senocak, A., Oh, T. H., Kim, J., Yang, M. H., & Kweon, I. S. (2018). Learning to localize sound source in visual scenes. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition 

[4] Valverde, F. R., Hurtado, J. V., & Valada, A. (2021). There is more than meets the eye: Self-supervised multi-object detection and tracking with sound by distilling multimodal knowledge. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition

[5] Zhou, J., Wang, J., Zhang, J., Sun, W., Zhang, J., Birchfield, S., ... & Zhong, Y. (2022, October). Audio–Visual Segmentation. In Computer Vision–ECCV 2022: 17th European Conference

[6] Ziegler, A., & Asano, Y. M. (2022). Self-supervised learning of object parts for semantic segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition
