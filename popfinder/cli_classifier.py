import argparse
import os
from popfinder.dataloader import GeneticData
from popfinder.classifier import PopClassifier

def load_classifier(arg_output_folder):

    classifier_name = "classifier.pkl"
    output_folder = arg_output_folder if arg_output_folder is not None else os.getcwd()
    classifier = PopClassifier.load(os.path.join(output_folder, classifier_name))

    return classifier


def main():

    parser = argparse.ArgumentParser(
        prog="pop_classifier",
        description="PopClassifier: A tool for population assignment from" + 
        " genetic data using classification neural networks."
    )

    # Arguments for determining which function to use
    parser.add_argument('--load_data', action='store_true', help='Load data from genetic and sample data files.')
    parser.add_argument('--train', action='store_true', help='Train a neural network.')
    parser.add_argument('--test', action='store_true', help='Test a trained neural network.')
    parser.add_argument('--assign', action='store_true', help='Assign samples to populations.')
    parser.add_argument('--rank_site_importance', action='store_true', help='Rank sites by importance for classification.')
    parser.add_argument('--plot_training_curve', action='store_true', help='Plot training curve.')
    parser.add_argument('--plot_confusion_matrix', action='store_true', help='Plot confusion matrix.')
    parser.add_argument('--plot_structure', action='store_true', help='Plot structure of population assignment.')
    parser.add_argument('--plot_assignment', action='store_true', help='Plot assignment of samples to populations.')

    # Arguments for loading data
    parser.add_argument('--genetic_data', type=str, default=None, help='Path to genetic data file.')
    parser.add_argument('--sample_data', type=str, default=None, help='Path to sample data file.')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of samples to use for testing.')
    parser.add_argument('--seed', type=int, default=123, help='Random seed for splitting data into training and testing sets.')
    parser.add_argument('--output_folder', type=str, default=None, help='Path to output folder. (Default: current directory)')

    # Arguments for training
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train for.')
    parser.add_argument('--valid_size', type=float, default=0.2, help='Proportion of training data to use for validation.')
    parser.add_argument('--cv_splits', type=int, default=1, help='Number of cross-validation splits.')
    parser.add_argument('--cv_reps', type=int, default=1, help='Number of cross-validation repetitions.')

    # Arguments for plotting structure/assignment
    parser.add_argument('--col_scheme', type=str, default="Spectral", help='Color scheme for plotting structure/assignment. (Default: "Spectral")')

    args = parser.parse_args()

    if args.load_data:
        data = GeneticData(args.genetic_data, args.sample_data, args.test_size, args.seed)
        classifier = PopClassifier(data, output_folder=args.output_folder)
        classifier.save()

    elif args.train:
        classifier = load_classifier(args.output_folder)
        classifier.train(args.epochs, args.valid_size, args.cv_splits, args.cv_reps)
        classifier.save()

    elif args.test:
        classifier = load_classifier(args.output_folder)
        classifier.test()
        classifier.get_classification_summary(save=True)
        classifier.save()

    elif args.assign:
        classifier = load_classifier(args.output_folder)
        classifier.assign_unknown(save=True)
        classifier.save()

    elif args.rank_site_importance:
        classifier = load_classifier(args.output_folder)
        classifier.rank_site_importance()
        classifier.save()

    elif args.plot_training_curve:
        classifier = load_classifier(args.output_folder)
        classifier.plot_training_curve()

    elif args.plot_confusion_matrix:
        classifier = load_classifier(args.output_folder)
        classifier.plot_confusion_matrix()

    elif args.plot_structure:
        classifier = load_classifier(args.output_folder)
        classifier.plot_structure(args.col_scheme)

    elif args.plot_assignment:
        classifier = load_classifier(args.output_folder)
        classifier.plot_assignment(args.col_scheme)

    else:         
        print("No function selected. Use --help for more information")
    

if __name__ == "__main__":
    main()
    
