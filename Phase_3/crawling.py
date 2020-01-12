import scrapy
from time import sleep
import json
import random
import numpy as np


class spider1(scrapy.Spider):
    name = "spider1"
    start_urls = [
        'https://www.semanticscholar.org/paper/Tackling-Climate-Change-with-Machine-Learning-Rolnick-Donti/998039a4876edc440e0cabb0bc42239b0eb29644',
        'https://www.semanticscholar.org/paper/Sublinear-Algorithms-for-(%CE%94%2B-1)-Vertex-Coloring-Assadi-Chen/eb4e84b8a65a21efa904b6c30ed9555278077dd3',
        'https://www.semanticscholar.org/paper/Processing-Data-Where-It-Makes-Sense%3A-Enabling-Mutlu-Ghose/4f17bd15a6f86730ac2207167ccf36ec9e6c2391']
    id = -1
    n = 5000
    def parse(self, response):
        self.id += 1
        NAME_SELECTOR = 'title ::text'
        REFRENCE_SELECTOR = "//a[@data-selenium-selector='title-link']/@href"
        YEAR_SELECTOR = "//span[@data-selenium-selector='paper-year']/span/span/text()"
        ABSTRACT_SELECTOR = "//div[has-class('text-truncator', 'abstract__text text--preline')]/text()"
        AUTHOR_SELECTOR = "//a[has-class('author-list__link', 'author-list__author-name')]/span/span/text()"
        print(len(self.start_urls))
        refrences = ["https://www.semanticscholar.org" + str(i) for i in response.xpath(REFRENCE_SELECTOR).getall()[0:10]]
        urls = []
        count = 0
        for url in refrences:
            if count >= 5:
                break
            if len(self.start_urls) <= self.n and url not in self.start_urls:
                self.start_urls.append(url)
                urls.append(url)
                count += 1
        with open("papers_data.json", "a") as filee:
            json.dump({
                'url': response.request.url,
                'id': self.id,
                'title': response.css(NAME_SELECTOR).extract_first(),
                'abstract': response.xpath(ABSTRACT_SELECTOR).extract_first(),
                'year': response.xpath(YEAR_SELECTOR).extract_first(),
                'author': response.xpath(AUTHOR_SELECTOR).extract(),
                'references': refrences,
            }, filee, indent=1)
            filee.write(',\n')
        for url in urls:
            yield response.follow(url, self.parse)
