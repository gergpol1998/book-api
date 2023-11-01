# HOW TO RUN 

docker-compose up -d

# API ENDPOIN POST METHOD
### contentbase-filtering
http://localhost:5002/recommendation

### json raw body
{
    "titles": ["BOOK-6610001664", "BOOK-6610000001"]
}

### collaborative-filtering
http://localhost:5001/recommendation

### json raw body
{
    "book_names": ["Initial D First Stage"]
}
