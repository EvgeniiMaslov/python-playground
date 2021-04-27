import requests
import urllib
from bs4 import BeautifulSoup

import os
import time

from io import BytesIO

import pandas as pd
from datetime import date, timedelta

import telegram_send



class CFG:
    
    data_path = 'C:\\Users\\sqrte\\avito-scrap\\data'

    url = 'https://www.avito.ru/shebekino/doma_dachi_kottedzhi/prodam/dom-ASgBAgICAkSUA9AQ2AjOWQ'

    today = date.today().strftime("%d_%m_%Y")
    yersterday = date.today() - timedelta(days=1)
    yersterday = yersterday.strftime("%d_%m_%Y")

    
    content_tag = 'div'
    content_atr = 'class'
    content_atr_value = 'iva-item-root-G3n7v photo-slider-slider-3tEix iva-item-list-2_PpT iva-item-redesign-1OBTh items-item-1Hoqq items-listItem-11orH js-catalog-item-enum'
    
    id_tag = 'div'
    id_atr = 'data-marker'
    id_atr_value = 'item'
    
    href_tag = 'a'
    href_atr = 'data-marker'
    href_atr_value = 'item-title'
    
    title_tag = 'h3'
    title_atr = 'class'
    title_atr_value = 'title-root-395AQ iva-item-title-1Rmmj title-listRedesign-3RaU2 title-root_maxHeight-3obWc text-text-1PdBw text-size-s-1PUdo text-bold-3R9dt'
    
    price_tag = 'meta'
    price_atr = 'itemprop'
    price_atr_value = 'price'
    
    geo_tag = 'span'
    geo_atr = 'class'
    geo_atr_value = 'geo-address-9QndR text-text-1PdBw text-size-s-1PUdo'



def download_html(url, user_agent='wswp', num_retries=2):

    print('Downloading:', url)
    
    headers = {'User-agent': user_agent}
    request = urllib.request.Request(url, headers=headers)
    
    try:
        html = urllib.request.urlopen(request).read().decode('utf-8')
    
    except urllib.error.URLError as e:
        print(f'Download error: {e}')
        html = None
        
        if num_retries > 0:
            if hasattr(e, 'code') and 500 <= e.code < 600:
                return download(url, num_retries - 1)
    
    return html




def get_number_of_pages(html):
    soup = BeautifulSoup(html, 'html.parser')
    span = soup.find_all('span', attrs={'class':'pagination-item-1WyVp'})
    pages = span[-2].contents[0]
    return int(pages)



def get_content_from_html(html):
    soup = BeautifulSoup(html, 'html.parser')
    divs = soup.find_all(CFG.content_tag, attrs={CFG.content_atr:CFG.content_atr_value})
    div_cnt = len(divs)
    
    ravel_div = ''
    for div in divs:
        ravel_div += str(div)
        
    return ravel_div, div_cnt



def download_pages(url, sleep_time=5):
    
    first_page = download_html(url)
    n = get_number_of_pages(first_page)
    
    pages, ads_count = get_content_from_html(first_page)
    
    for page_num in range(2, n+1):

        time.sleep(sleep_time)

        page_url = url + '?p={}'.format(page_num)
        
        html = download_html(page_url)
        content, cnt = get_content_from_html(html)

        pages += content
        ads_count += cnt
    
    return pages, ads_count


def scrap_pages(html):
    soup = BeautifulSoup(html, 'html.parser')
    
    all_ids = soup.find_all(CFG.id_tag, attrs={CFG.id_atr:CFG.id_atr_value})
    data_item_ids = [int(id['data-item-id']) for id in all_ids]
    ids = [str(id['id']) for id in all_ids]
    
    hrefs = soup.find_all(CFG.href_tag, attrs={CFG.href_atr:CFG.href_atr_value})
    hrefs = ['https://www.avito.ru'+str(href['href']) for href in hrefs]
    
    descriptions = soup.find_all(CFG.title_tag, attrs={CFG.title_atr:CFG.title_atr_value})
    descriptions = [desc.contents[0] for desc in descriptions]
    
    prices = soup.find_all(CFG.price_tag, attrs={CFG.price_atr:CFG.price_atr_value})
    prices = [int(price['content']) for price in prices]
    
    geos = soup.find_all(CFG.geo_tag, attrs={CFG.geo_atr : CFG.geo_atr_value})
    geos = [str(geo.contents)[7:-8] for geo in geos]
    
    
    return data_item_ids, ids, hrefs, descriptions, prices, geos


def make_dataframe(data_item_ids, ids, hrefs, descriptions, prices, geos):
    df = pd.DataFrame()
    df['data_item_id'] = data_item_ids
    df['id'] = ids
    df['href'] = hrefs
    df['description'] = descriptions
    df['price'] = prices
    df['geo'] = geos
    df['data_item_id'] = data_item_ids
    df['date'] = CFG.today
    return df

def check_new_sold_houses():
    df_new = pd.read_csv(os.path.join(CFG.data_path, 'shebekino_houses_{}.csv'.format(CFG.today)))

    
    df_old = pd.read_csv(os.path.join(CFG.data_path,'shebekino_houses_{}.csv'.format(CFG.yersterday)))

    df = pd.concat([df_new, df_old], axis=0)
    not_dup = df.drop_duplicates(subset=['data_item_id', 'id', 'href', 'description', 'geo'], keep=False)
    
    new = not_dup[not_dup['date'] == CFG.today]
    sold = not_dup[not_dup['date'] == CFG.yersterday]
    
    return new, sold

    
    

def send_to_telegram(new, sold, ads_cnt):
    total_new = new.shape[0]
    total_sold = sold.shape[0]

    messages = []

    basic_info = 'Сегодня доступно {} объявлений о продаже дома.\nЗа прошедшие сутки появилось {} новых объявлений о\
                продаже дома;\n{} домов было продано/снято с продажи'.format(ads_cnt, total_new, total_sold)
    
    messages.append(basic_info)

    for new_house_idx in range(new.shape[0]):
        message = 'Новый дом на продажу: {},\n{} - {} рублей,\n{}'.format(new.iloc[new_house_idx].href,
                                                                          new.iloc[new_house_idx].description,
                                                                          new.iloc[new_house_idx].price,
                                                                          new.iloc[new_house_idx].geo)
        messages.append(message)


    for sold_house_idx in range(sold.shape[0]):
        message = 'За прошедшие сутки продали: {},\n{} - {} рублей,\n{}'.format(sold.iloc[sold_house_idx].href,
                                                                                sold.iloc[sold_house_idx].description,
                                                                                sold.iloc[sold_house_idx].price,
                                                                                sold.iloc[sold_house_idx].geo)
        messages.append(message)


    telegram_send.send(messages=messages)


def main():

    print('Downloading html pages...')
    pages, ads_cnt = download_pages(CFG.url)
    print('Collecting usefull information...')
    data_item_ids, ids, hrefs, descriptions, prices, geos = scrap_pages(pages)
    
    df_new = make_dataframe(data_item_ids, ids, hrefs, descriptions, prices, geos)

    if not os.path.isdir(CFG.data_path):
        os.mkdir(CFG.data_path)

    filepath = os.path.join(CFG.data_path, 'shebekino_houses_{}.csv'.format(CFG.today))
    print('Saving inforamtion to: {}'.format(filepath))
    df_new.to_csv(filepath, index=False)

    print('Check new/sold houses...')
    new, sold = check_new_sold_houses()
    print('Sending info to telegram...')
    send_to_telegram(new, sold, ads_cnt)



if __name__ == '__main__':
    main()