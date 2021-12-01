#coding: UTF-8

import slackweb
import sys

def postToSlack(text):
  # url = "https://hooks.slack.com/services/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"
  url= "https://hooks.slack.com/services/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"
  slack = slackweb.Slack(url=url)
  slack.notify(text=text)

if __name__ == "__main__":
  text = "this is test!"
  msg = sys.argv[1]
  postToSlack(msg)