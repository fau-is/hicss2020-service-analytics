import keras
import csv
import distance

try:
    from itertools import izip as zip
except ImportError:
    pass
from jellyfish._jellyfish import damerau_levenshtein_distance
import util


def test(args, preprocess_manager):
    batch_size = args.batch_size_test
    result_dir = args.result_dir
    task = args.task

    if preprocess_manager.num_features_additional > 0:
        lines, caseids, lines_add, sequence_max_length, num_features_all, num_features_activities = preprocess_manager.create_test_set()
    else:
        lines, caseids, sequence_max_length, num_features_all, num_features_activities = preprocess_manager.create_test_set()

    model = keras.models.load_model(
        '%smodel_%s.h5' % (args.checkpoint_dir, preprocess_manager.iteration_cross_validation))

    predict_size = 1
    data_set_name = args.data_set.split('.csv')[0]
    generic_result_dir = result_dir + data_set_name + "__" + task
    fold_result_dir = generic_result_dir + "_%d%s" % (preprocess_manager.iteration_cross_validation, ".csv")
    result_dir = fold_result_dir

    with open(result_dir, 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(
            ["CaseID", "Prefix length", "Groud truth", "Predicted", "Levenshtein", "Damerau", "Jaccard"])

        for prefix_size in range(2, sequence_max_length):
            util.llprint("\nPrefix size: %d\n" % prefix_size)

            # if additional attributes exists
            if preprocess_manager.num_features_additional > 0:

                for line, caseid, line_add in zip(lines, caseids, lines_add):

                    cropped_line = ''.join(line[:prefix_size])
                    cropped_line_add = line_add[:prefix_size]

                    if '!' in cropped_line:
                        continue

                    ground_truth = ''.join(line[prefix_size:prefix_size + predict_size])
                    predicted = ''

                    for i in range(predict_size):

                        if len(ground_truth) <= i:
                            continue

                        input_vec, num_features_all, num_features_activities = preprocess_manager.encode_test_set_add(
                            args, cropped_line, cropped_line_add, batch_size)
                        y = model.predict(input_vec, verbose=0)
                        y_char = y[0][:]
                        prediction = preprocess_manager.getSymbol(y_char)
                        cropped_line += prediction
                        predicted += prediction

                        if prediction == '!':
                            print('! predicted, end case')
                            break

                    output = []
                    if len(ground_truth) > 0:

                        output.append(caseid)
                        output.append(prefix_size)
                        output.append(str(ground_truth).encode("utf-8"))
                        output.append(str(predicted).encode("utf-8"))
                        output.append(1 - distance.nlevenshtein(predicted, ground_truth))

                        dls = 1 - (damerau_levenshtein_distance(str(predicted), str(ground_truth)) / max(len(predicted),
                                                                                                         len(
                                                                                                             ground_truth)))
                        if dls < 0:
                            dls = 0
                        output.append(dls)
                        output.append(1 - distance.jaccard(predicted, ground_truth))
                        spamwriter.writerow(output)


            # if no additional attributes exists            
            else:
                for line, caseid in zip(lines, caseids):

                    cropped_line = ''.join(line[:prefix_size])

                    if '!' in cropped_line:
                        continue

                    ground_truth = ''.join(line[prefix_size:prefix_size + predict_size])
                    predicted = ''

                    for i in range(predict_size):

                        if len(ground_truth) <= i:
                            continue

                        input_vec = preprocess_manager.encode_test_set(cropped_line, batch_size)
                        y = model.predict(input_vec, verbose=0)
                        y_char = y[0][:]
                        prediction = preprocess_manager.getSymbol(y_char)
                        cropped_line += prediction
                        predicted += prediction

                        if prediction == '!':
                            print('! predicted, end case')
                            break

                    output = []
                    if len(ground_truth) > 0:

                        output.append(caseid)
                        output.append(prefix_size)
                        output.append(str(ground_truth).encode("utf-8"))
                        output.append(str(predicted).encode("utf-8"))
                        output.append(1 - distance.nlevenshtein(predicted, ground_truth))

                        dls = 1 - (damerau_levenshtein_distance(str(predicted), str(ground_truth)) / max(len(predicted),
                                                                                                         len(
                                                                                                             ground_truth)))
                        if dls < 0:
                            dls = 0
                        output.append(dls)
                        output.append(1 - distance.jaccard(predicted, ground_truth))
                        spamwriter.writerow(output)
