"""
Code to create a train/validation and test set split.

The split is stratified so that we ensure equal numbers of
the following important features in the training, validation
and test sets

- Documents one or more parts names
- Documents with two or more part names (required for pairs of parts)
- Documents with one or more feature names
- Documents with feature names and part names 
- Documents with descriptions
"""

import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def warn_if_file_exists(pathname, force_overwrite):
    if pathname.exists():
        print(f"Warning!! {pathname} already exists.")
        if not force_overwrite:
            print("Use --force_overwrite to force regenerate it")
            sys.exit(1)


def save_train_validation_test_file(
        pathname,
        train,
        validation,
        test,
        force_overwrite
):
    """
    Save file containing a given train, validation and test 
    set
    """
    warn_if_file_exists(pathname, force_overwrite)
    print(pathname)
    print(f"Total size {len(train) + len(validation) + len(test)}")
    print(f"Train size {len(train)}")
    print(f"Validation size {len(validation)}")
    print(f"Test size {len(test)}")
    print(" ")
    assert len(set(train)) == len(train), "Duplicate document ids??"
    assert len(set(validation)) == len(validation), "Duplicate document ids??"
    assert len(set(test)) == len(test), "Duplicate document ids??"
    assert len(set(test).union(set(validation)).union(set(train))) == \
           len(train) + len(validation) + len(test), "Duplicate document ids??"
    data = {
        "train": train,
        "validation": validation,
        "test": test
    }
    with open(pathname, 'w', encoding='utf8') as fp:
        json.dump(data, fp, indent=4, ensure_ascii=False, sort_keys=False)


def save_split_for_features(
        filename,
        output,
        document_guids,
        features,
        feature_index,
        train,
        validation,
        test,
        force_overwrite):
    """
    Save a file with the split for a given feature set
    """
    train = set(train)
    validation = set(validation)
    test = set(test)
    train_subset = []
    validation_subset = []
    test_subset = []
    assert len(document_guids) == len(set(document_guids)), "Duplicate document guids"
    for document_index, document_guid in enumerate(document_guids):
        if features[document_index, feature_index]:
            found = False
            if document_guid in train:
                train_subset.append(document_guid)
                found = True
            if document_guid in validation:
                validation_subset.append(document_guid)
                found = True
            if document_guid in test:
                test_subset.append(document_guid)
                found = True
            assert found, "Document is on in train validation or test set??"

    pathname = output / filename
    save_train_validation_test_file(
        pathname,
        train_subset,
        validation_subset,
        test_subset,
        force_overwrite
    )


def create_sub_split(document_guids, features, fraction_b):
    """
    Split the document guids based on the features into 
    two buckets a and b.   The sizes are proportional to
    1-fraction_b and fraction_b
    """
    assert fraction_b > 0.0, "Must split by non-zero fraction"
    assert fraction_b < 1.0, "Must split by faction smaller than 1"
    sss = StratifiedShuffleSplit(n_splits=1, test_size=fraction_b)
    n_samples = len(document_guids)
    splits = sss.split(np.zeros(n_samples), features)
    splits = list(splits)
    assert len(splits) == 1, "Should have just 1 split"
    set_a, set_b = splits[0]
    assert len(set(set_a)) == len(set_a), "Duplicate indices"
    assert len(set(set_b)) == len(set_b), "Duplicate indices"
    assert len(set(set_a).intersection(set(set_b))) == 0, "Duplicate indices"
    document_guids_a = [document_guids[i] for i in set_a]
    document_guids_b = [document_guids[i] for i in set_b]
    features_b = features[set_b]
    return document_guids_a, document_guids_b, features_b


def create_main_split(document_guids, features, options):
    validation_fraction = options.validation_fraction
    test_fraction = options.test_fraction
    val_test_fraction = validation_fraction + test_fraction

    # First separate the train set from the validation and test set
    train_document_guids, val_test_document_guids, val_test_features = \
        create_sub_split(document_guids, features, val_test_fraction)

    second_split_fraction = test_fraction / val_test_fraction
    validation_document_guids, test_document_guids, test_features = \
        create_sub_split(val_test_document_guids, val_test_features, second_split_fraction)

    return train_document_guids, validation_document_guids, test_document_guids


def create_feature_lists(all_data):
    """
    Create an np.array with the features which we want to 
    split equally in there
    """
    document_guids = []
    num_documents = len(all_data)

    features = np.zeros((num_documents, 5), dtype=np.int32)
    for index, document_guid in enumerate(all_data):
        document_guids.append(document_guid)

        doc_data = all_data[document_guid]
        # Documents one or more parts names
        if len(doc_data["body_names"]) > 0:
            features[index, 0] = 1

        # Documents with two or more part names (required for pairs of parts)
        if len(doc_data["body_names"]) >= 2:
            features[index, 1] = 1

        # Documents with one or more feature names
        if len(doc_data["feature_names"]) > 0:
            features[index, 2] = 1

        # Documents with one or more part names and one or more feature names
        if len(doc_data["body_names"]) > 0 and len(doc_data["feature_names"]) > 0:
            features[index, 3] = 1

        assert doc_data["document_name"] != "", "Should have a document name"

        # Documents with descriptions 
        if len(doc_data["document_description"]) >= 2:
            features[index, 4] = 1

    assert len(document_guids) == len(set(document_guids)), "Duplicate document guids"

    return document_guids, features


def create_split(input, output, options):
    """
    Main function to create the split
    """

    # Load the raw data
    with open(input, "r") as fp:
        all_data = json.load(fp)

    print(f"Number of documents in {input} is {len(all_data)}")
    print(" ")

    # Create the features
    document_guids, features = create_feature_lists(all_data)

    # Create the stratified split
    train, validation, test = create_main_split(document_guids, features, options)

    # Save the full train/validation/test split file
    force_overwrite = options.force_overwrite
    save_train_validation_test_file(
        output / "train_val_test_split.json",
        train,
        validation,
        test,
        force_overwrite
    )

    # Save documents one or more parts names
    save_split_for_features(
        "train_val_test_partnames.json",
        output,
        document_guids,
        features,
        0,
        train,
        validation,
        test,
        force_overwrite
    )

    # Save documents with two or more part names (required for pairs of parts)
    save_split_for_features(
        "train_val_test_two_or_more_partnames.json",
        output,
        document_guids,
        features,
        1,
        train,
        validation,
        test,
        force_overwrite
    )

    # Save documents with one or more feature names
    save_split_for_features(
        "train_val_test_featurenames.json",
        output,
        document_guids,
        features,
        2,
        train,
        validation,
        test,
        force_overwrite
    )

    save_split_for_features(
        "train_val_test_partnames_and_featurenames.json",
        output,
        document_guids,
        features,
        3,
        train,
        validation,
        test,
        force_overwrite
    )

    save_split_for_features(
        "train_val_test_descriptions.json",
        output,
        document_guids,
        features,
        4,
        train,
        validation,
        test,
        force_overwrite
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Input abc_text_files_nnn.json file')
    parser.add_argument('--output', type=str, required=True, help='Output folder')
    parser.add_argument('--force_overwrite', action='store_true', help='Allow existing files to be overwritten')
    parser.add_argument('--validation_fraction', type=float, default=0.15,
                        help='Fraction of documents in validation set')
    parser.add_argument('--test_fraction', type=float, default=0.15, help='Fraction of documents in test set')
    args = parser.parse_args()

    input = Path(args.input)
    output = Path(args.output)
    create_split(input, output, args)
