from textblob import TextBlob

def interpret_command(input_text):
    blob = TextBlob(input_text)

    if 'create ticket' in blob.lower():
        action = 'create_ticket'
        params = {"system": "Remedy"}
        if 'high priority' in blob.lower():
            params["ticket_details"] = {"priority": "High"}

        # Here we're extracting a noun phrase as the ticket description, for simplicity
        if blob.noun_phrases:
            params["ticket_details"]["description"] = str(blob.noun_phrases[0])
    else:
        action = None
        params = None

    return action, params

def main():
    # For instance, if TextBlob interpreted the user's input text and determined that the user wants to create a new ticket...
    input_text = "I need to create a high priority ticket regarding a network outage in San Francisco."
    action, parameters = interpret_command(input_text)
    if action and parameters:
        execute_action_based_on_text(action, parameters)
    else:
        print("Sorry, I couldn't understand your request.")

if __name__ == "__main__":
    main()