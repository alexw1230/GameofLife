from datetime import datetime, timedelta


def check_quest_reminder(filename="quests.txt"):
    now = datetime.now()

    with open(filename, "r") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            try:
                # Split quest info and exp value
                quest_info, exp_str = line.split(",", 1)
                exp_value = int(exp_str.strip())
                
                # Split only on the first colon
                name, times = quest_info.split(":", 1)
                start_str, end_str = times.strip().split("-")

                # Convert to datetime
                start_time = datetime.strptime(start_str.strip(), "%H:%M").replace(
                    year=now.year, month=now.month, day=now.day
                )
                end_time = datetime.strptime(end_str.strip(), "%H:%M").replace(
                    year=now.year, month=now.month, day=now.day
                )

                # Handle overnight meetings
                if end_time <= start_time:
                    end_time += timedelta(days=1)

                # Current meeting
                if start_time <= now <= end_time:
                    return f"Current quest: {name.strip()}", exp_value

                # 30-minute reminder
                if now <= start_time <= now + timedelta(minutes=30):
                    return f"Get to this quest: {name.strip()}", exp_value

            except ValueError:
                continue

    return None, 0
