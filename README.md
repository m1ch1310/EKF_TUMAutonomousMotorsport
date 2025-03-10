# EKF_TUMAutonomousMotorsport
This repository is about optimizing an Extended-Kalmanfilter used for autonomous racing.
Their are plenty of different python-implementations of EKFs. The excel with the corresponding name includes the output-data of said implementation.


the indexes ".0". ".1", ... describe the different values for the process-noise-matrix.
"vd"       => velocity in d-direction is included in the state-vector.
"dynamicH" => Measurement-noise-vector is changed based on distance to detectin and velocity
"turnrate" => turnrate is included in the state-vector.
