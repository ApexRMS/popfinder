import argparse
from popfinder.dataloader import GeneticData
from popfinder.classifier import PopClassifier

def main():

    parser = argparse.ArgumentParser(
        prog="pop_classifier",
        description="PopClassifier: A tool for population assignment from" + 
        " genetic data using classification neural networks."
    )

    # Arguments for determining which function to use
    parser.add_argument('--load_data', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--assign', action='store_true')
    parser.add_argument('--plot_training_curve', action='store_true')
    parser.add_argument('--plot_confusion_matrix', action='store_true')
    parser.add_argument('--plot_structure', action='store_true')
    parser.add_argument('--plot_assignment', action='store_true')

    # Arguments for loading data
    parser.add_argument('--genetic_data', type=str, default=None)
    parser.add_argument('--sample_data', type=str, default=None)
    parser.add_argument('--test_size', type=float, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--output_folder', type=str, default=None)

    # Arguments for training
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--valid_size', type=float, default=None)
    parser.add_argument('--cv_splits', type=int, default=None)
    parser.add_argument('--cv_reps', type=int, default=None)

    # Arguments for plotting structure/assignment
    parser.add_argument('--col_scheme', type=str, default=None)

    args = parser.parse_args()

    if args.load_data:
        data = GeneticData(args.genetic_data, args.sample_data, args.test_size, args.seed)
        classifier = PopClassifier(data, args.output_folder)
        classifier.save()

    if args.train:
        classifier = PopClassifier.load(data, args.seed, args.output_folder)
        classifier.train(args.epochs, args.valid_size, args.cv_splits, args.cv_reps)
        classifier.save()

    if args.test:
        classifier = PopClassifier.load(args.output_folder)
        classifier.test()
        classifier.save()

    if args.assign:
        classifier = PopClassifier.load(args.output_folder)
        classifier.assign()
        classifier.save()

    if args.plot_training_curve:
        classifier = PopClassifier.load(args.output_folder)
        classifier.plot_training_curve()

    if args.plot_confusion_matrix:
        classifier = PopClassifier.load(args.output_folder)
        classifier.plot_confusion_matrix()

    if args.plot_structure:
        classifier = PopClassifier.load(args.output_folder)
        classifier.plot_structure(args.col_scheme)

    if args.plot_assignment:
        classifier = PopClassifier.load(args.output_folder)
        classifier.plot_assignment(args.col_scheme)

if __name__ == "__main__":
    main()
    
