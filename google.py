import json
import re
from string import ascii_lowercase, digits
from time import sleep
from urllib import parse

import requests
from bs4 import BeautifulSoup

import Deropy.scraping as scr


class Google:
    def __init__(self):
        self.SEARCH_URL = 'https://www.google.co.jp/search'
        self.SUGGEST_URL = 'http://www.google.co.jp/complete/search?output=toolbar&ie=utf-8&oe=utf-8&client=firefox&q='
        self.ARAMAKIJAKE_URL = 'http://aramakijake.jp/keyword/export.php?keyword='
        self.session = requests.session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:57.0) Gecko/20100101 Firefox/57.0'})

    def Search(self, keyword, type='text', maximum=100):
        ''' Google検索 '''
        print('Google', type.capitalize(), 'Search :', keyword)
        result, total = [], 0
        query = self.query_gen(keyword, type)

        if maximum == 0:
            return

        while True:
            # 検索
            html = self.session.get(next(query)).text
            links = self.get_links(html, type)

            # 検索結果の追加
            if not len(links):
                print('-> No more links')
                break
            elif len(links) > maximum - total:
                result += links[:maximum - total]
                break
            else:
                result += links
                total += len(links)
            sleep(0.5)

        print('-> Finally got', str(len(result)), 'links')
        return result

    def Suggest(self, keyword, jap=False, alph=False, num=False):
        ''' サジェスト取得 '''
        # 文字リスト作成
        chars = ['', ' ']
        chars += [' ' + chr(i) for i in range(12353, 12436)] if jap else []
        chars += [' ' + char for char in ascii_lowercase] if alph else []
        chars += [' ' + char for char in digits] if num else []

        # サジェスト取得
        suggests = {}
        for char in chars:
            print('\'' + keyword + char + '\'')
            res = self.session.get(self.SUGGEST_URL + keyword + char)
            suggests[char if char == '' else char[-1]] = res.json()[1]
            sleep(0.5)
        return suggests

    def Value(self, keywords):
        ''' 検索回数取得 '''
        # リスト化
        if not isinstance(keywords, list):
            keywords = [keywords]

        # 検索回数取得
        vals = {}
        for keyword in keywords:
            url = self.ARAMAKIJAKE_URL + keyword
            content = self.session.get(url).content
            content = scr.encode_bytes(content)
            # print(content)
            val = re.split('[,\n]', content)[5]
            vals[keyword] = int(val) if str.isdecimal(val) else 0
            sleep(0.5)
        return vals

    def query_gen(self, keyword, type):
        ''' 検索クエリジェネレータ '''
        page = 0
        while True:
            if type == 'text':
                params = parse.urlencode({
                    'q': keyword,
                    'num': '100',
                    'filter': '0',
                    'start': str(page * 100)})
            elif type == 'image':
                params = parse.urlencode({
                    'q': keyword,
                    'tbm': 'isch',
                    'filter': '0',
                    'ijn': str(page)})

            # print(self.SEARCH_URL + '?' + params)
            yield self.SEARCH_URL + '?' + params
            page += 1

    def get_links(self, html, type):
        ''' html内のリンクを取得 '''
        soup = BeautifulSoup(html, 'lxml')
        if type == 'text':
            elements = soup.select('.rc > .r > a')
            links = [e['href'] for e in elements]
        elif type == 'image':
            elements = soup.select('.rg_meta.notranslate')
            jsons = [json.loads(e.get_text()) for e in elements]
            links = [js['ou'] for js in jsons]
        return links


if __name__ == '__main__':
    google = Google()
    # result = google.Search('ドラえもん', type='image', maximum=200)
    # result = google.Search('ドラえもん', type='text', maximum=200)
    # print(result)

    suggests = google.Suggest('ドラえもん')
    # suggests = google.Suggest('ドラえもん', jap=True, alph=True, num=True)
    print(suggests)

    n = 0
    for key in suggests.keys():
        for suggest in suggests[key]:
            result = google.Search(suggest, type='text', maximum=10)
            n += 1
            print(n, len(result))
            sleep(0.5)
