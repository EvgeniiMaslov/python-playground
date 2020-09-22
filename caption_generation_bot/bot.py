#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 11:11:58 2020

@author: evgenii
"""

import logging
import model
from model import apply_model_to_image_raw_bytes

from telegram import (ReplyKeyboardMarkup, ReplyKeyboardRemove)
from telegram.ext import (Updater, CommandHandler, MessageHandler, Filters,
                          ConversationHandler)

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


def photo(update, context):
    user = update.message.from_user
    photo_file = update.message.photo[-1].get_file()
    photo_file.download('user_photo.png')
    answer = apply_model_to_image_raw_bytes(open("user_photo.png", "rb").read())
    logger.info("Photo of %s: %s", user.first_name, 'user_photo.png')
    update.message.reply_text(answer)



def text(update, context):
    update.message.reply_text('I dont understand you, please send me a photo')

def main():
    updater = Updater("YOUR TOKEN", use_context=True)
    dp = updater.dispatcher
    conv_handler = MessageHandler(Filters.photo, photo)
    dp.add_handler(conv_handler)
    text_handler = MessageHandler(Filters.text, text)
    dp.add_handler(text_handler)
    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
