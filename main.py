import requests


def test_recommendation():

    print('Hey LVMH, welcome to this simple meta recommender system!')
    print('The default recommendations are songs.')
    print('Maybe outfits in the future? :)')
    print('\n')
    print('Alright, let\'s start!')
    print('Remember: this is some form of cold start for rec-sys')
    print('That means you\'re not supposed to be recommended good songs right away')

    res = requests.get("http://127.0.0.1:8000/recommend")
    assert res.status_code == 200, f"Api Error {res.status_code}"
    print(f'First song recommended: {res.json()["recommendation"]}')

    while True:
        print('Did you enjoy it? Answer Yes or No.\n')
        feedback = input()
        try:
            assert feedback.lower() in ('yes', 'no')
        except AssertionError:
            print('Invalid feedback: Answer Yes or No!\n')
            continue
        res = requests.post("http://127.0.0.1:8000/feedback",
                            json={'feedback': True if feedback.lower() == 'yes' else False})
        assert res.status_code == 200, f'Api Error {res.status_code}'
        print('Thanks for the feedback!')
        print('Do you want to be recommended another song? Answer Yes or No!\n')
        another_song = input()
        while True:
            try:
                assert another_song.lower() in ('yes', 'no')
                break
            except AssertionError:
                print('Invalid feedback: Answer Yes or No!')
        if another_song.lower() == 'yes':
            res = requests.get("http://127.0.0.1:8000/recommend")
            assert res.status_code == 200, f'Api Error {res.status_code}'
            print(f'Next song recommended: {res.json()["recommendation"]}')
        else:
            print('Alright, bye!')
            return


if __name__ == "__main__":
    test_recommendation()
