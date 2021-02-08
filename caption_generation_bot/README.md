# Caption generation bot

What can he do:

1. Caption generation via neural network (CNN encoder + RNN decoder)



How to use:

1. Clone repository
2. Get telegram api token:
  * write the following commands to the t.me/BotFather - /start -> /newbot
  * choose a name and username to your bot
  * BotFather should send you a token, something like "1151332045:AAHU6CVXZ0kwsRAIXJOmOIXctYBFw0SWUi0"
3. In a line below 'YOUR TOKEN HERE' should be replaced by the API token you received
```
updater = Updater("YOUR TOKEN HERE", use_context=True)
```
4. Execute the code and send image to the bot