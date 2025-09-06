from book import Book

class Library:
    def __init__(self, name: str, address: str):
        self.name = name
        self.address = address
        self.books = {}

    def __repr__(self):
        return ""

    def get_info(self):
        print("========")
        print(f"Libray name: {self.name}")
        print(f"Libray address: {self.address}")

    def add_book(self, book: Book):
        self.books[book.id] = book

    def get_all_books(self):
        for book_id, book in self.books.items():
            book.get_info()