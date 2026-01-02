def next_birthday(date, birthdays):
    '''
    Find the next birthday after the given date.

    @param:
    date - a tuple of two integers specifying (month, day)
    birthdays - a dict mapping from date tuples to lists of names, for example,
      birthdays[(1,10)] = list of all people with birthdays on January 10.

    @return:
    birthday - the next day, after given date, on which somebody has a birthday
    list_of_names - list of all people with birthdays on that date
    '''
    month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    def next_day(m, d):
        if d < month_days[m - 1]:
            return (m, d + 1)
        if m < 12:
            return (m + 1, 1)
        return (1, 1)

    m, d = date
    current = next_day(m, d)  

    for _ in range(366):
        if current in birthdays and birthdays[current]:
            return current, birthdays[current]
        current = next_day(*current)

    return (1, 1), []