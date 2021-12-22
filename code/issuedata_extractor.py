import argparse
import csv
import json
from pprint import pprint

import pandas as pd

from jira import JIRA, JIRAError

APACHE_JIRA_SERVER = 'https://issues.apache.org/jira/'


# Read csv items into a dict
def read_csv(path):
    dt = pd.read_csv(path, index_col=0, skiprows=0).T.to_dict()
    return dt


def key_dic(keys, jira):
    dictionary = {'key': 'value'}

    for original_key in keys:
        issue_list = jira.search_issues('key in (' + original_key + ')',
                                        fields='key, description',
                                        maxResults=1000)

        for issue in issue_list:
            dictionary[issue.key] = original_key

    return dictionary


def get_issue_var(fields, field, name):
    if hasattr(fields, name) and field is not None:
        return field
    return ''


def main():
    # Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default=None)
    args = parser.parse_args()

    # Read the issue keys into a dict
    dt = read_csv(args.csv_path)

    # Authenticate with the jira server
    jira = JIRA(APACHE_JIRA_SERVER, basic_auth=('Mohamed_Soliman',
                                                'Smbam@2005'))
    # Store the issue keys in a comma separated string format
    keys_dic = key_dic(dt.keys(), jira)
    keys_str = ','.join(dt.keys())

    # Obtain the issue list
    issue_list = jira.search_issues('key in (' + keys_str + ')',
                                    fields="key, parent, summary, description,"
                                           "attachment, comment, issuelinks, "
                                           "issuetype, labels, priority, "
                                           "resolution, status, subtasks, "
                                           "votes, watches",
                                    maxResults=1000)

    json_list = []
    for issue in issue_list:
        if hasattr(issue.fields, 'parent'):
            parent_var = issue.fields.parent
        else:
            parent_var = ''
        attcounter = 0
        if hasattr(issue.fields, 'attachment'):
            attach_var = issue.fields.attachment
            for attach in attach_var:
                if "pdf" in attach.filename or "doc" in attach.filename:
                    attcounter = attcounter + (attach.size / 1024)

        if hasattr(issue.fields, 'summary') and issue.fields.summary is \
                not None:
            summary_var = issue.fields.summary
        else:
            summary_var = ''

        if hasattr(issue.fields,
                   'description') and issue.fields.description is not None:
            desc_var = issue.fields.description
        else:
            desc_var = ""

        comments_size = 0
        comments_text = ''
        if hasattr(issue.fields,
                   'comment') and issue.fields.comment is not None:
            comments_var = issue.fields.comment.comments
            for comm_var in comments_var:
                if "cassandraqa" in comm_var.author.name or "hudson" in \
                        comm_var.author.name or "Diff:" in comm_var.body:
                    continue
                else:
                    comments_size = comments_size + len(comm_var.body)
                    comments_text += '\n' + comm_var.body
                    # print(comm_var.author.name)

        children_issue_list = jira.search_issues('parent=' + issue.key,
                                                 fields="key,parent,"
                                                        "description,"
                                                        "attachment,"
                                                        "comment",
                                                 maxResults=1000)

        attcounter_children = 0
        desc_var_children = 0
        comments_size_children = 0
        for child_issue in children_issue_list:

            if hasattr(child_issue.fields, 'attachment'):
                attach_var = child_issue.fields.attachment
                for attach in attach_var:
                    if "pdf" in attach.filename or "txt" in \
                            attach.filename or "doc" in attach.filename:
                        attcounter_children = attcounter_children + (
                                attach.size / 1024)

            if hasattr(child_issue.fields,
                       'description') and child_issue.fields.description \
                    is not None:
                desc_var_child = child_issue.fields.description
                desc_var_children = desc_var_children + len(desc_var_child)

            if hasattr(child_issue.fields,
                       'comment') and child_issue.fields.comment is not \
                    None:
                comments_var = child_issue.fields.comment.comments
                for comm_var in comments_var:
                    comments_size_children = comments_size_children + len(
                        comm_var.body)

        original_key = keys_dic[issue.key]

        print(issue.key)
        # print('{},{},{},{},{},{},{},{},{}'.format(issue.key, parent_var,
        #                                           desc_var,
        #                                           comments_text,
        #                                           attcounter,
        #                                           comments_size,
        #                                           desc_var_children,
        #                                           attcounter_children,
        #                                           comments_size_children))
        fields = issue.fields

        dictionary = {
            'key': issue.key,
            'parent': parent_var,
            'summary': summary_var,
            'description': desc_var,
            'comments': comments_text,
            '#_attachments': attcounter,
            'comment_size': comments_size,
            'issuelinks': str(get_issue_var(fields, fields.issuelinks,
                                            'issuelinks')),
            'issuetype': str(get_issue_var(fields, fields.issuetype,
                                           'issuetype')),
            'labels': get_issue_var(fields, fields.labels, 'labels'),
            'priority': str(get_issue_var(fields, fields.priority,
                                       'priority')),
            'resolution': str(get_issue_var(fields, fields.resolution,
                                        'resolution')),
            'status': str(get_issue_var(fields, fields.status, 'status')),
            'subtasks': get_issue_var(fields, fields.subtasks, 'subtasks'),
            'votes': str(get_issue_var(fields, fields.votes, 'votes')),
            'watches': str(get_issue_var(fields, fields.watches,
                                         'watches')),
            'description_children': desc_var_children,
            '#_attachements_children': attcounter_children,
            'comment_size_children': comments_size_children
        }

        json_list.append(dictionary)

    with open('output.json', 'w') as json_file:
        json.dump(json_list, json_file, indent=4)


if __name__ == '__main__':
    main()
