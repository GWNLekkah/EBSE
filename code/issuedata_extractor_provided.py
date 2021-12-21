import argparse
import csv
import pandas as pd
import re

from jira import JIRA, JIRAError

APACHE_JIRA_SERVER = 'https://issues.apache.org/jira/'

def read_csv(p):
    """
    Load data from csv file

    :return:
    """
    #df = pd.read_csv(p)
    #keys = df.iloc[:, 0]
    #a2a = df.iloc[:, 1]
    dt = pd.read_csv(p, index_col=0, skiprows=0).T.to_dict()
    return dt


def key_dic(keys,jira):
    
    dictionary = {'key':'value'}
    
    for originalKey in keys:
         issue_list = jira.search_issues('key in (' + originalKey + ')',fields="key,description", maxResults=1000)
         
         for issue in issue_list:      
               dictionary[issue.key] = originalKey
    return dictionary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default=None)
    #parser.add_argument("--number_of_clusters", type=int, default=8)
    args = parser.parse_args()

    dt = read_csv(args.csv_path)

    jira = JIRA(APACHE_JIRA_SERVER, basic_auth=('Mohamed_Soliman', 'Smbam@2005'))
    
    keys_dic = key_dic(dt.keys(),jira)

    #print(keys_dic)
    
    #print(dt)
    #print(dt.keys())
    keys_str = ','.join(dt.keys())

    #list_urls = read_csv(args.csv_path)
    
    #issue_list = jira.search_issues('project=TAJO', maxResults=3000)
    issue_list = jira.search_issues('key in (' + keys_str + ')',fields="key,parent,description,attachment,comment", maxResults=1000)
    #issue_list = jira.search_issues('parent=TAJO-1118',fields="key,parent,description,attachment")
    #issue_list = jira.search_issues('key=TAJO-1118',fields="key,parent,description,attachment",maxResults=3000)
    
    with open('issues.csv', 'w', newline='') as csvfile:
         issueswriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
         
         issueswriter.writerow(['key','parent key','description size','attachment size','comments size','A2A','children description size','children attachment size','children comments size'])
         
         for issue in issue_list:
              if hasattr(issue.fields,'parent'):     
                   parent_var = issue.fields.parent
              else:
                   parent_var = ""
              attcounter = 0 
              if hasattr(issue.fields,'attachment'): 
                   attach_var = issue.fields.attachment
                   for attach in attach_var:
                         if "pdf" in attach.filename or "doc" in attach.filename:
                                attcounter = attcounter + (attach.size / 1024)
         
              if hasattr(issue.fields,'description') and issue.fields.description is not None: 
                   desc_var = issue.fields.description
              else:
                   desc_var = ""
         
              comments_size = 0
              if hasattr(issue.fields,'comment') and issue.fields.comment is not None: 
                   comments_var = issue.fields.comment.comments
                   for comm_var in comments_var:
                         if "cassandraqa" in comm_var.author.name or "hudson" in comm_var.author.name or "Diff:" in comm_var.body:
                               continue
                         else:
                               comments_size = comments_size + len(comm_var.body)
                               #print(comm_var.author.name)
              
              
              children_issue_list = jira.search_issues('parent='+issue.key,fields="key,parent,description,attachment,comment", maxResults=1000)
              
              attcounter_children = 0
              desc_var_children = 0
              comments_size_children = 0
              for child_issue in children_issue_list:
              
                   if hasattr(child_issue.fields,'attachment'): 
                         attach_var = child_issue.fields.attachment
                         for attach in attach_var:
                                if "pdf" in attach.filename or "txt" in attach.filename or "doc" in attach.filename:
                                       attcounter_children = attcounter_children + (attach.size / 1024)
         
                   if hasattr(child_issue.fields,'description') and child_issue.fields.description is not None: 
                         desc_var_child = child_issue.fields.description
                         desc_var_children = desc_var_children + len(desc_var_child)
              
                   if hasattr(child_issue.fields,'comment') and child_issue.fields.comment is not None: 
                         comments_var = child_issue.fields.comment.comments
                         for comm_var in comments_var:
                                comments_size_children = comments_size_children + len(comm_var.body)
              
               
              original_key = keys_dic[issue.key]
              
              print('{},{},{},{},{},{},{},{},{}'.format(issue.key, parent_var, len(desc_var),attcounter,comments_size,dt[original_key]['a2a'],desc_var_children,attcounter_children,comments_size_children))
              issueswriter.writerow([issue.key,parent_var,len(desc_var),attcounter,comments_size,dt[original_key]['a2a'],desc_var_children,attcounter_children,comments_size_children])
    
    #print('Test result:')
    #print('{} -> {}'.format(label, doc))

    # cluster_importance(mgp)


if __name__ == '__main__':
    main()
