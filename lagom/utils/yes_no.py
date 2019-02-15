def ask_yes_or_no(msg):
    r"""Ask user to enter yes or no to a given message. 
    
    Args:
        msg (str): a message
    """
    print(msg)
    
    while True:
        answer = str(input('>>> ')).lower().strip()
        
        if answer[0] == 'y':
            return True
        elif answer[0] == 'n':
            return False
        else:
            print("Please answer 'yes' or 'no':")
