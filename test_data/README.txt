MotionSense test videos — all from the official UCF101 TEST split (testlist01).
The model never saw these videos (or these actors) during training or model selection.

file  ->  ground-truth activity
JumpingJack_g02_c03.mp4  ->  Jumping Jack
JumpingJack_g05_c03.mp4  ->  Jumping Jack
JumpingJack_g01_c03.mp4  ->  Jumping Jack
PushUps_g05_c03.mp4  ->  Push-Ups
PushUps_g04_c05.mp4  ->  Push-Ups
PushUps_g01_c04.mp4  ->  Push-Ups
BodyWeightSquats_g01_c01.mp4  ->  Bodyweight Squats
BodyWeightSquats_g05_c04.mp4  ->  Bodyweight Squats
BodyWeightSquats_g06_c01.mp4  ->  Bodyweight Squats
Lunges_g01_c04.mp4  ->  Lunges
Lunges_g06_c04.mp4  ->  Lunges
Lunges_g07_c04.mp4  ->  Lunges
TaiChi_g07_c01.mp4  ->  Tai Chi
TaiChi_g06_c01.mp4  ->  Tai Chi
TaiChi_g03_c01.mp4  ->  Tai Chi
BoxingSpeedBag_g07_c02.mp4  ->  Boxing (Speed Bag)
BoxingSpeedBag_g04_c07.mp4  ->  Boxing (Speed Bag)
BoxingSpeedBag_g06_c03.mp4  ->  Boxing (Speed Bag)

mixed_activities.mp4 (34 s) contains, in order:
    0.0s –   4.0s  Jumping Jack
    4.0s –  10.0s  Push-Ups
   10.0s –  16.0s  Bodyweight Squats
   16.0s –  22.0s  Lunges
   22.0s –  28.0s  Tai Chi
   28.0s –  34.0s  Boxing (Speed Bag)

Note: mixed_activities.mp4 is a demonstration of the activity timeline, not an evaluation.
Its Speed Bag segment uses clip g06 because the longest Speed Bag clip (g07) is one the model
misclassifies. The single-activity files above are NOT filtered: they include 5 clips the model
gets wrong (13/18 correct), consistent with the 72.9% official test accuracy.
