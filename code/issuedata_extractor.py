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
def get_issue_var(fields, name, is_int=False):
    if hasattr(fields, name) and getattr(fields, name) is not None:
        return getattr(fields, name)
    if is_int:
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
        attcounter = 0
        if hasattr(issue.fields, 'attachment'):
            attach_var = issue.fields.attachment
            for attach in attach_var:
                if "pdf" in attach.filename or "doc" in attach.filename:
                    attcounter = attcounter + (attach.size / 1024)

        # Get the comment count and text
        comments_size = 0
        comments_count = 0
        comments_text = []
        if hasattr(issue.fields,
                   'comment') and issue.fields.comment is not None:
            comments_var = issue.fields.comment.comments
            for comm_var in comments_var:
                if "cassandraqa" in comm_var.author.name or "hudson" in \
                        comm_var.author.name or "Diff:" in comm_var.body:
                    continue
                else:
                    comments_size = comments_size + len(comm_var.body)
                    comments_count += 1
                    comments_text.append(comm_var.body)

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

        attcounter_children = 0
        desc_var_children = 0
        comments_size_children = 0
        for child_issue in children_issue_list:
            # Get the attachments size
            if hasattr(child_issue.fields, 'attachment'):
                attach_var = child_issue.fields.attachment
                for attach in attach_var:
                    if "pdf" in attach.filename or "txt" in \
                            attach.filename or "doc" in attach.filename:
                        attcounter_children = attcounter_children + (
                                attach.size / 1024)

            # Get the description size
            if hasattr(child_issue.fields,
                       'description') and child_issue.fields.description \
                    is not None:
                desc_var_child = child_issue.fields.description
                desc_var_children = desc_var_children + len(desc_var_child)

            # Get the comment size
            if hasattr(child_issue.fields,
                       'comment') and child_issue.fields.comment is not \
                    None:
                comments_var = child_issue.fields.comment.comments
                for comm_var in comments_var:
                    comments_size_children = comments_size_children + len(
                        comm_var.body)

        # Printing to give user info about the progress
        print('  ' + issue.key)

        fields = issue.fields

        # Create a dict and store it in the json list
        dictionary = {
            'key': issue.key,
            'parent': str(get_issue_var(fields, 'parent')),
            'summary': get_issue_var(fields, 'summary'),
            'description': get_issue_var(fields, 'description'),
            'comments': comments_text,
            '#_attachments': attcounter,
            'comments_count': comments_count,
            'issuelinks': len(get_issue_var(fields, 'issuelinks')),
            'issuetype': str(get_issue_var(fields, 'issuetype')),
            'labels': get_issue_var(fields, 'labels'),
            'priority': str(get_issue_var(fields, 'priority')),
            'resolution': str(get_issue_var(fields, 'resolution')),
            'status': str(get_issue_var(fields, 'status')),
            'subtasks': len(get_issue_var(fields, 'subtasks')),
            'votes': int(str(get_issue_var(fields, 'votes', is_int=True))),
            'watch_count': watch_count_var,
            'description_children': desc_var_children,
            '#_attachements_children': attcounter_children,
            'comment_size_children': comments_size_children
        }
        json_list.append(dictionary)

    with open('output.json', 'w') as json_file:
        json.dump(json_list, json_file, indent=4)


if __name__ == '__main__':
    main()
