""" these dictionaries are used for plotting functions. Currently only
for the line plots in plot_lineplots.py but it will be useful to keep colors,
markers, and other style elements listed here consistent across other plots. """


experiment_stimulus_name_dict = {
    1 : "Motion",
    2 : "Discr. Motion",
    3 : "Bijection",
    4 : "Appearance", 
    5 : "Static Targets",
    "5b" : "Solo Targets"
}
model_cam_name_dict = {
    "eigen_cam" : "EigenCAM",
    "grad_cam" : "GradCAM",
    "grad_cam_plusplus" : "GradCAM++",
    "cam_plusplus" : "GradCAM++", 
    "i3d_rgb": "I3D",
    "i3dFalsergb": "I3D",
    "i3d_nln" : "NLN",
    "i3dTruergb": "NLN",
    "slowfast_fast" : "SlowFast Fast",
    "slowfastFalsefast": "SlowFast Fast",
    "slowfast_slow" : "SlowFast Slow",
    "slowfastFalseslow": "SlowFast Slow",
}

multi_model_color_dict = {
    "i3dFalsergb" : "#AC5EF0", # I3d is purple
    "i3dTruergb" : "#BD075F", # NLN is pink
    "slowfastFalsefast" : "#FBB705", # Fast is yellow
    "slowfastFalseslow" : "#02A1A5",  # Slow is teal
}

vivid_experiment_color_dict = {
    1 : "#0072B2", # dark blue
    2 : "#12C34F", #green
    3 : "#F3BB0B", # yellow
    4 : "#E62B62",  # fuschia
    5 : "#94027A", # dark purple
    "5b" : "#56C8E9" # light blue
}
pastel_experiment_color_dict = {
    # RBGA hex values use alpha = 0.3
    1 : ("#62B6E44d"), # dark blue
    2 : ("#85E8984d"), # green
    3 : ("#F7F1A74d"), # yellow
    4 : ("#FFBBCC4d"), # pink
    5 : ("#C5C1F34d"), # light purple
    "5b" : ("#94D8EC4d")# light blue
}

label_color_dict = {
    "circle": ("#82e3e8"),
    "line": ("#74bcdb"),
    "quadrilateral": ("#d1f0e9"),
    "sinusoid": ("#d1eef0"),
    "spiral": ("#d1d8f0"),
    "triangle": ("#6dc2ae"),
    "zigzag": ("#8faeeb"),
    "Cat": ("#f7c1c1"),
    "Cattle": ("#fcc0b1"),
    "Fish": ("#fccfac"),
    "Flower": ("#fcacc7"),
    "Motorcycle": ("#f0afd7"),
    "Train": ("#f59fe8"),
    "Truck": ("#dbabc6")
}

label_marker_dict = {
    "circle": "x",
    "line": "o",
    "quadrilateral": "v",
    "sinusoid": "^",
    "spiral": "s",
    "triangle": "D",
    "zigzag": "h",
    "Cat": "x",
    "Cattle": "o",
    "Fish": "v",
    "Flower": "^",
    "Motorcycle": "s",
    "Train": "D",
    "Truck": "h"
}

# dictionary of absolute maximum of metric across all CAM and models
cleaned_metric_dict = {'kl_div_1_pre_softmax': 7.035656107244919,
 'iou_1_pre_softmax': 0.2293559305863784,
 'pearson_1_pre_softmax': 5.984556658297634e-06,
 'mse_1_pre_softmax': 0.18324503864892436,
 'covariance_1_pre_softmax': 0.0033476120351479564,
 'precision_1_pre_softmax': 0.3690542145404286,
 'recall_1_pre_softmax': 1.0,
 'kl_div_1_post_softmax': 3.19724894127214,
 'iou_1_post_softmax': 0.21140675665905245,
 'pearson_1_post_softmax': 0.0003918476884032921,
 'mse_1_post_softmax': 0.2637779866463797,
 'covariance_1_post_softmax': 0.003234556411187054,
 'precision_1_post_softmax': 0.27689039364011797,
 'recall_1_post_softmax': 1.0,
 'kl_div_2_pre_softmax': 9.439739646501016,
 'iou_2_pre_softmax': 0.17204435921228484,
 'pearson_2_pre_softmax': 5.385998705043918e-06,
 'mse_2_pre_softmax': 0.18701039318086715,
 'covariance_2_pre_softmax': 0.002201245960990435,
 'precision_2_pre_softmax': 0.227263564922881,
 'recall_2_pre_softmax': 0.9407197111591371,
 'kl_div_2_post_softmax': 3.8478945116191814,
 'iou_2_post_softmax': 0.1654554337928647,
 'pearson_2_post_softmax': 0.00010977447604246125,
 'mse_2_post_softmax': 0.15545827663987466,
 'covariance_2_post_softmax': 0.002152407257149793,
 'precision_2_post_softmax': 0.21914447269144485,
 'recall_2_post_softmax': 0.9549331022659512,
 'kl_div_3_pre_softmax': 4.263908222865258,
 'iou_3_pre_softmax': 0.22656338926880468,
 'pearson_3_pre_softmax': 6.661749446216575e-06,
 'mse_3_pre_softmax': 0.060959850648502706,
 'covariance_3_pre_softmax': 0.003050043592065885,
 'precision_3_pre_softmax': 0.3002356669974934,
 'recall_3_pre_softmax': 0.9725274258661166,
 'kl_div_3_post_softmax': 6.529284261858769,
 'iou_3_post_softmax': 0.19104072643506026,
 'pearson_3_post_softmax': 0.00018938765417563197,
 'mse_3_post_softmax': 0.04553083188033143,
 'covariance_3_post_softmax': 0.0024204905577918227,
 'precision_3_post_softmax': 0.25407301969938384,
 'recall_3_post_softmax': 1.0,
 'kl_div_4_pre_softmax': 5.41798514850815,
 'iou_4_pre_softmax': 0.2625434064451517,
 'pearson_4_pre_softmax': 7.470192326193211e-06,
 'mse_4_pre_softmax': 0.05950425646423308,
 'covariance_4_pre_softmax': 0.0020817616541472252,
 'precision_4_pre_softmax': 0.38717367385190476,
 'recall_4_pre_softmax': 0.6642431163984627,
 'kl_div_4_post_softmax': 3.5388993806392066,
 'iou_4_post_softmax': 0.20520017933969648,
 'pearson_4_post_softmax': 0.00014381403372225137,
 'mse_4_post_softmax': 0.0322179353300184,
 'covariance_4_post_softmax': 0.0016767388149155043,
 'precision_4_post_softmax': 0.3514779124595773,
 'recall_4_post_softmax': 0.9072433192686357,
 'kl_div_5_pre_softmax': 10.196139007729755,
 'iou_5_pre_softmax': 0.220054695636876,
 'pearson_5_pre_softmax': 7.762328266667548e-06,
 'mse_5_pre_softmax': 0.042068160138253416,
 'covariance_5_pre_softmax': 0.0030577964754668666,
 'precision_5_pre_softmax': 0.2786835693423155,
 'recall_5_pre_softmax': 0.9728091915155823,
 'kl_div_5_post_softmax': 9.559964788545631,
 'iou_5_post_softmax': 0.22396769234419414,
 'pearson_5_post_softmax': 4.737965902534215e-05,
 'mse_5_post_softmax': 0.03990665877796796,
 'covariance_5_post_softmax': 0.0030663945363231175,
 'precision_5_post_softmax': 0.275090288273044,
 'recall_5_post_softmax': 0.9606579162614013,
 'kl_div_5b_pre_softmax': 7.185282550931246,
 'iou_5b_pre_softmax': 0.25201544295468986,
 'pearson_5b_pre_softmax': 9.646821945878151e-06,
 'mse_5b_pre_softmax': 0.11928374824358842,
 'covariance_5b_pre_softmax': 0.0036542927404916116,
 'precision_5b_pre_softmax': 0.43766576997495726,
 'recall_5b_pre_softmax': 0.9959794914049029,
 'kl_div_5b_post_softmax': 7.185282550931246,
 'iou_5b_post_softmax': 0.24813590082732048,
 'pearson_5b_post_softmax': 0.00012971551751824088,
 'mse_5b_post_softmax': 0.11928374824358842,
 'covariance_5b_post_softmax': 0.003607487639106988,
 'precision_5b_post_softmax': 0.4354284754785913,
 'recall_5b_post_softmax': 0.9946167559129488,
 'activation_1_pre_softmax': 89.94815963745117,
 'activation_1_post_softmax': 16.170370483398436,
 'activation_2_pre_softmax': 81.79341462787829,
 'activation_2_post_softmax': 20.72054732473273,
 'activation_3_pre_softmax': 52.77462348159479,
 'activation_3_post_softmax': 21.61646340110085,
 'activation_4_pre_softmax': 28.10983488831339,
 'activation_4_post_softmax': 3.3719471364781475,
 'activation_5_pre_softmax': 28.28434909119898,
 'activation_5_post_softmax': 26.347219038982782,
 'activation_5b_pre_softmax': 44.96758355034722,
 'activation_5b_post_softmax': 44.96758355034722}

class_id_marker_dict = {
    0 : "x",
    1 : "o",
    2 : "v",
    3 : "^",
    4 : "<", 
    5 : ">",
    6 : "s",
    7 : "D",
    8 : "h"
}