import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def send_email():
    fromaddr = "kkgdcx@outlook.com"
    toaddr = "o115sama@hotmail.com"  # يمكنك تعيين البريد الإلكتروني المستلم هنا
    msg = MIMEMultipart()
    msg['From'] = fromaddr
    msg['To'] = toaddr
    msg['Subject'] = "Alert: Malicious Behavior Detected"

    body = "A malicious behavior was detected. Check the attached video."
    msg.attach(MIMEText(body, 'plain'))

    # Use the Outlook SMTP server settings
    server = smtplib.SMTP('smtp.office365.com', 587)
    server.starttls()
    
    # Log in to the server using your Outlook credentials
    server.login(fromaddr, "Aa@12345678")
    
    # Convert the message to string and send the email
    text = msg.as_string()
    server.sendmail(fromaddr, toaddr, text)
    
    # Quit the server after sending the email
    server.quit()
