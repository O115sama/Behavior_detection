import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

def send_email(image_path, detected_action):
    fromaddr = "441100***@tvtc.edu.sa"  
    toaddr = "443330***@tvtc.edu.sa"  
    msg = MIMEMultipart()
    msg['From'] = fromaddr
    msg['To'] = toaddr
    msg['Subject'] = "تنبيه: سلوك مشبوه تم اكتشافه"

    action_messages = {
        'slapping': "تم اكتشاف سلوك صفع.",
        'drop kicking': "تم اكتشاف سلوك ركل.",
        'punching person': "تم اكتشاف سلوك لكم.",
        'covering_face': "تم اكتشاف سلوك تغطية الوجه.",
        'falling': "تم اكتشاف سلوك سقوط.",
        'waving': "تم اكتشاف سلوك ت waved.",
        'smoking hookah': "تم اكتشاف سلوك تدخين الشيشة.",
        'smoking': "تم اكتشاف سلوك تدخين.",
    }

    body = action_messages.get(detected_action, "تم اكتشاف سلوك مشبوه. تحقق من الصورة المرفقة.")
    msg.attach(MIMEText(body, 'plain'))

    with open(image_path, "rb") as attachment:
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f'attachment; filename={image_path}')
        msg.attach(part)


    server = smtplib.SMTP('smtp.office365.com', 587)
    server.starttls()
    
    server.login(fromaddr, "********") 
    text = msg.as_string()
    
    server.sendmail(fromaddr, toaddr, text)

    server.quit()
