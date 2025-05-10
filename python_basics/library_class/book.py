class Book:
    def __init__(self, id: int, name: str, author: str, edition: int, publisher: str):
        self.id = id
        self.name = name
        self.author = author
        self.edition = edition
        self.publisher = publisher

    def get_info(self):
        print("===========")
        print(f"Book name: {self.name}")
        print(f"Book author: {self.author}")
        print(f"Book edition: {self.edition}")
        print(f"Book publisher: {self.publisher}")
        print("===========")
        
    