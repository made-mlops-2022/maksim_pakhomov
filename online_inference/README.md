Commands to run rest service:

```docker pull averagename/rest_service```   

Then create .env file, which will contain id of google drive file, in my case it will be

```ID=1b7heC46hjKrAz4DftH1FO8PYYmp_06Vs```

To start rest service:   

```docker run --env-file .env -p 80:80 averagename/rest_service```

If you want to run request from client, use

```python3 app/client.py```   

as example