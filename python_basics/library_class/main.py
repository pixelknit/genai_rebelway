from library import Library
from book import Book

#books
book1 = Book(1, "book1", "author1", 2, "Publisher1")
book2 = Book(2, "book2", "author2", 1, "Publisher2")

library1 = Library("Lib1","Soho London")

library1.get_info()


library1.add_book(book1)
library1.add_book(book2)

library1.get_all_books()