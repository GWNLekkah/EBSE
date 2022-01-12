import argparse
import json

import pandas as pd

from jira import JIRA, JIRAError

APACHE_JIRA_SERVER = 'https://issues.apache.org/jira/'


# Read csv items into a dict
def read_csv(path):
    dt = pd.read_csv(path, index_col=0, skiprows=0).T.to_dict()
    return dt


# Get issue metadata (field)
def get_issue_var(fields, name, field_type='str'):
    if hasattr(fields, name) and getattr(fields, name) is not None:
        if field_type == 'bool':
            return 1
        return getattr(fields, name)
    if field_type == 'int' or field_type == 'bool':
        return 0
    return ''


def main():
    # Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default=None)
    args = parser.parse_args()

    # Read the issue keys into a dict
    print('read issue keys from csv')
    dt = read_csv(args.csv_path)

    # Authenticate with the jira server
    print('athenticate with server')
    jira = JIRA(APACHE_JIRA_SERVER, basic_auth=('Mohamed_Soliman',
                                                'Smbam@2005'))
    # Store the issue keys in a comma separated string format
    keys_str = ','.join(dt.keys())

    # Obtain the issue list (single issue at a time to keep requests small)
    print('searching issues')
    issue_list = []
    for key in dt.keys():
        # Printing to give user updates about the progress
        print('  ' + key)
        issue = jira.search_issues('key=' + key,
                                   fields="key, parent, summary, description,"
                                          "attachment, comment, issuelinks, "
                                          "issuetype, labels, priority, "
                                          "resolution, status, subtasks, "
                                          "votes, watches")
        issue_list.extend(issue)

    print('processing issues')
    json_list = []
    for issue in issue_list:
        # Get the attachment count
        attachment_count = 0
        if hasattr(issue.fields, 'attachment'):
            for attach in issue.fields.attachment:
                if "pdf" in attach.filename or "doc" in attach.filename:
                    attachment_count += 1

        # Get the comment count and text
        comment_list = []
        if hasattr(issue.fields,
                   'comment') and issue.fields.comment is not None:
            comments_var = issue.fields.comment.comments
            for comm_var in comments_var:
                if "cassandraqa" in comm_var.author.name or "hudson" in \
                        comm_var.author.name or "Diff:" in comm_var.body:
                    continue
                else:
                    comment_list.append(comm_var.body)

        # Get the number of watchers
        if hasattr(issue.fields,
                   'watches') and issue.fields.watches is not None:
            watch_count_var = issue.fields.watches.watchCount
        else:
            watch_count_var = 0

        # Get information about the children
        children_issue_list = jira.search_issues('parent=' + issue.key,
                                                 fields="key,parent,"
                                                        "description,"
                                                        "attachment,"
                                                        "comment")
        children_count = len(children_issue_list)

        # Printing to give user info about the progress
        print('  ' + issue.key)

        fields = issue.fields

        # Create a dict and store it in the json list
        dictionary = {
            'key': issue.key,
            'summary': get_issue_var(fields, 'summary'),
            'summary_length': len(get_issue_var(fields, 'summary')),
            'description': get_issue_var(fields, 'description'),
            'description_length': len(get_issue_var(fields, 'description')),
            'comment_list': comment_list,
            'comment_length': sum([len(comment) for comment in comment_list]),
            '#_comments': len(comment_list),
            '#_attachments': attachment_count,
            '#_issuelinks': len(get_issue_var(fields, 'issuelinks')),
            'issuetype': str(get_issue_var(fields, 'issuetype')),
            'labels': get_issue_var(fields, 'labels'),
            'priority': str(get_issue_var(fields, 'priority')),
            'resolution': str(get_issue_var(fields, 'resolution')),
            '#_subtasks': len(get_issue_var(fields, 'subtasks')),
            '#_votes': int(str(get_issue_var(fields, 'votes', 'int'))),
            '#_watches': watch_count_var,
            'has_parent': get_issue_var(fields, 'parent', 'bool'),
            '#_children': children_count
        }
        json_list.append(dictionary)

    with open('output.json', 'w+') as json_file:
        json.dump(json_list, json_file, indent=4)


if __name__ == '__main__':
    main()
