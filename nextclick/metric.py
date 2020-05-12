import csv
import sklearn
import util


def calc_metrics(args):
    prefix = 0
    prefix_all_enabled = 1
    prediction = list()
    gt_label = list()

    result_dir = args.result_dir
    if args.cross_validation == False:
        result_dir_fold = result_dir + args.data_set.split(".csv")[0] + "__" + args.task + "_0.csv"
    else:
        result_dir_fold = result_dir + args.data_set.split(".csv")[
            0] + "__" + args.task + "_%d" % args.iteration_cross_validation + ".csv"

    with open(result_dir_fold, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        next(reader)

        for row in reader:
            if row == []:
                continue
            else:
                if int(row[1]) == prefix or prefix_all_enabled == 1:
                    gt_label.append(row[2])
                    prediction.append(row[3])

    util.llprint("\n\n")
    util.llprint("Metrics:\n")
    util.llprint("Accuracy: %f\n" % sklearn.metrics.accuracy_score(gt_label, prediction))

    # calc metric for each label, and find their weighted mean
    util.llprint(
        "Precision (weighted): %f\n" % sklearn.metrics.precision_score(gt_label, prediction, average='weighted'))
    util.llprint("Recall (weighted): %f\n" % sklearn.metrics.recall_score(gt_label, prediction, average='weighted'))
    util.llprint("F1-Score (weighted): %f\n" % sklearn.metrics.f1_score(gt_label, prediction, average='weighted'))

    # calc macro metric for each label, and find their unweighted mean
    util.llprint("Precision (macro): %f\n" % sklearn.metrics.precision_score(gt_label, prediction, average='macro'))
    util.llprint("Recall (macro): %f\n" % sklearn.metrics.recall_score(gt_label, prediction, average='macro'))
    util.llprint("F1-Score (macro): %f\n" % sklearn.metrics.f1_score(gt_label, prediction, average='macro'))

    # calc micro metric over all examples
    util.llprint("Precision (micro): %f\n" % sklearn.metrics.precision_score(gt_label, prediction, average='micro'))
    util.llprint("Recall (micro): %f\n" % sklearn.metrics.recall_score(gt_label, prediction, average='micro'))
    util.llprint("F1-Score (micro): %f\n\n" % sklearn.metrics.f1_score(gt_label, prediction, average='micro'))

    return sklearn.metrics.accuracy_score(gt_label, prediction), sklearn.metrics.precision_score(gt_label, prediction,
                                                                                                 average='weighted'), sklearn.metrics.recall_score(
        gt_label, prediction, average='weighted'), sklearn.metrics.f1_score(gt_label, prediction, average='weighted')
