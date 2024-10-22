from morph_classify import ClassifyMorph, ClassifyStat
import os, sys

# basic setup
main_path = "/Users/dzliu/Desktop/morph"
input_path = os.path.join(main_path, "morph_data")
output_path = os.path.join(main_path, "output")

train_set_name = "train.fits"
test_set_name = "test.fits"
out_set_name = "xoutput.fits"

# initialize classification class
morph_obj = ClassifyMorph(train_set_name, test_set_name, out_set_name, input_path, input_path, output_path)
test_set = morph_obj.load_test_set()
train_set, label_set = morph_obj.load_train_set()
pred_out, pred_id = morph_obj.classify(train_set, test_set, label_set)
morph_obj.write_set(pred_out, pred_id)
morph_obj.stats_plot()

# feature importance
print("Estimate feature importance and confusion matrix")
stats_obj = ClassifyStat(train_set_name, input_path)
stats_obj.stats_plot()

