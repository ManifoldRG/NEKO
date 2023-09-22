from speed_test import main

@profile
def profile_main():
    main()

if __name__ == '__main__':
    profile_main()
