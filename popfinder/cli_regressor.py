import argparse
from popfinder.dataloader import GeneticData
from popfinder.regressor import PopRegressor

def main():

    parser = argparse.ArgumentParser(
        prog="pop_regressor",
        description="PopRegressor: A tool for population assignment from" +
        " genetic data using regression neural networks."
    )

    # Arguments for determining which function to use
    parser.add_argument('--load_data', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--assign', action='store_true')
    parser.add_argument('--plot_training_curve', action='store_true')
    parser.add_argument('--plot_location', action='store_true')
    parser.add_argument('--classify_by_contours', action='store_true')
    parser.add_argument('--plot_contour_map', action='store_true')
    parser.add_argument('--plot_confusion_matrix', action='store_true')
    parser.add_argument('--plot_structure', action='store_true')
    parser.add_argument('--plot_assignment', action='store_true')
    parser.add_argument('--get_assignment_summary', action='store_true')
    parser.add_argument('--get_classification_summary', action='store_true')

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

    # Arguments for contour classification
    parser.add_argument('--nboots', type=int, default=None)
    parser.add_argument('--num_contours', type=int, default=None)

    # Arguments for plotting structure/assignment
    parser.add_argument('--col_scheme', type=str, default=None)

    args = parser.parse_args()

    if args.load_data:
        data = GeneticData(args.genetic_data, args.sample_data, args.test_size, args.seed)
        regressor = PopRegressor(data, args.output_folder)
        regressor.save()

    if args.train:
        regressor = PopRegressor.load(args.output_folder)
        regressor.train(args.epochs, args.valid_size, args.cv_splits, args.cv_reps)
        regressor.save()
    
    if args.test:
        regressor = PopRegressor.load(args.output_folder)
        regressor.test()
        regressor.save()

    if args.assign:
        regressor = PopRegressor.load(args.output_folder)
        regressor.assign_unknown()
        regressor.save()

    if args.plot_training_curve:
        regressor = PopRegressor.load(args.output_folder)
        regressor.plot_training_curve()

    if args.plot_location:
        regressor = PopRegressor.load(args.output_folder)
        regressor.plot_location()

    if args.classify_by_contours:
        regressor = PopRegressor.load(args.output_folder)
        regressor.classify_by_contours(args.nboots, args.num_contours)
        regressor.save()

    if args.plot_contour_map:
        regressor = PopRegressor.load(args.output_folder)
        regressor.plot_contour_map()

    if args.plot_confusion_matrix:
        regressor = PopRegressor.load(args.output_folder)
        regressor.plot_confusion_matrix()

    if args.plot_structure:
        regressor = PopRegressor.load(args.output_folder)
        regressor.plot_structure(args.col_scheme)

    if args.plot_assignment:
        regressor = PopRegressor.load(args.output_folder)
        regressor.plot_assignment(args.col_scheme)

    if args.get_assignment_summary:
        regressor = PopRegressor.load(args.output_folder)
        regressor.get_assignment_summary()

    if args.get_classification_summary:
        regressor = PopRegressor.load(args.output_folder)
        regressor.get_classification_summary()

if __name__ == "__main__":
    main()