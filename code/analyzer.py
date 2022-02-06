# Auxiliary file for exploratory analysis of data

import collections
import json

import numpy
import matplotlib.pyplot as pyplot


features = {
    "summary_length": "interval",
    "description_length": "interval",
    "comment_length": "interval",
    "#_comments": "interval",
    "#_attachments": "interval",
    "#_issuelinks": "interval",
    "issuetype": "nominal",
    "labels": "nominal",
    "priority": "ordinal",
    "resolution": "nominal",
    "#_subtasks": "interval",
    "#_votes": "interval",
    "#_watches": "interval",
    "has_parent": "nominal",
    "#_children": "interval",
}


def plot(architectural_data, normal_data, feature, measurement_type):
    if measurement_type == 'interval':
        make_box_plot(architectural_data, normal_data, feature)
    elif measurement_type == 'nominal':
        make_frequency_plot(architectural_data, normal_data, feature)
    elif measurement_type == 'ordinal':
        make_frequency_plot(architectural_data, normal_data, feature)


def make_box_plot(architectural_data, normal_data, feature):
    fig, ax = pyplot.subplots()
    ax.boxplot([architectural_data, normal_data], showfliers=False)
    ax.set_title(feature)
    ax.set_xticklabels(['Architectural', 'Non-Architectural'])
    pyplot.show()
    fig.savefig(f'{feature}.png')


def make_frequency_plot(architectural_data, normal_data, feature):
    if feature == 'labels':
        return
    fig, [ax1, ax2] = pyplot.subplots(ncols=2)

    architectural_counts = collections.Counter(architectural_data)
    normal_counts = collections.Counter(normal_data)

    fields = set(architectural_counts) | set(normal_counts)
    x_labels = list(sorted(fields))
    x_points = list(range(len(x_labels)))

    architectural_heights = [architectural_counts[x] for x in x_labels]
    normal_heights = [normal_counts[x] for x in x_labels]

    ax1.barh(x_points, architectural_heights)
    ax1.set_title(f'{feature} - Architectural')
    ax1.set_yticks(x_points)
    ax1.set_yticklabels(x_labels)

    ax2.barh(x_points, normal_heights)
    ax2.set_title(f'{feature} - Non-Architectural')
    ax2.set_yticks(x_points)
    ax2.set_yticklabels(x_labels)

    pyplot.show()
    fig.savefig(f'{feature}.png')


def extract_feature(issues, feature_key):
    return [issue[feature_key] for issue in issues]


def main(architectural_issues, normal_issues):
    for feature_key, measurement_type in features.items():
        x = extract_feature(architectural_issues, feature_key)
        y = extract_feature(normal_issues, feature_key)
        plot(x, y, feature_key, measurement_type)


if __name__ == '__main__':
    print('Reading file 1')
    with open('./issuedata_extractor/architectural_issues.json') as file:
        data_1 = json.load(file)
    print('Reading file 2')
    with open('./issuedata_extractor/non_architectural_issues.json') as file:
        data_2 = json.load(file)
    print('Plotting')
    main(data_1, data_2)
