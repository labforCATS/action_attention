# before running, must set RESULTS_PATH to the value of 
#  _C.TEST.SAVE_RESULTS_PATH in slowfast/config/defaults.py
import pickle
# RESULTS_PATH = ".inprogress_i3d_preds"
RESULTS_PATH = "kinetics_preds_6_18"
# RESULTS_PATH = "last_500_kinetics_preds_6_18"

# options to toggle desired information
# TODO: should these also write files instead of just printing?
print_pred_categories = True
print_groundtruth = True
print_confidence_percentages = False

# convert from binary to human-readable data
input_file = open(RESULTS_PATH, "rb")
input_file = pickle.load(input_file)

# separate predictions from groundtruth data
# category is the prediction with highest confidence
pred_confidence = (input_file[0]).tolist()
pred_labels = []
for i in pred_confidence: 
    pred_labels.append(i.index(max(i)))

groundtruth_labels = (input_file[1]).tolist()

correct_count = 0
test_count = len(groundtruth_labels)
mislabels = []
for i in range(test_count):
    if (pred_labels[i] == groundtruth_labels[i]):
        correct_count = correct_count + 1
    else:
        mislabels.append(i)

print("The model correctly labeled ", correct_count, " out of ", test_count, " videos.")
#print("The model incorrectly labeled videos at (0-indexed) indices ", mislabels)

if(print_pred_categories):
    print("The model predicted: ", pred_labels[:10])
if(print_groundtruth):
    print("The true labels are: ", groundtruth_labels[:10])
if(print_confidence_percentages):
    print("The full list of predictions with confidence percentages is: ", pred_confidence[:1])

