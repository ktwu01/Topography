import math

def perp_pts(x, y, m, edge_length, edges):
    '''Create points perpendicular to the line segment at a set distance'''
    x0, y0, x1, y1 = edges
    if m == 0:
        # Flat edge cases
        if y0 == y1:
            y1 = y + edge_length
            y2 = y - edge_length
            x1 = x
            x2 = x
        if x0 == x1:
            y1 = y
            y2 = y
            x1 = x + edge_length
            x2 = x - edge_length
    else:
        # A line perpendicular to x-y will have slope (-1/m)
        m_perp = (-1 / m)

        # Use vector math to get points along perpendicular line
        if m > 0:
            x1 = x + (edge_length / math.sqrt(1 + m_perp ** 2))
            y1 = y + ((edge_length * m_perp) / math.sqrt(1 + m_perp ** 2))
            x2 = x - (edge_length / math.sqrt(1 + m_perp ** 2))
            y2 = y - ((edge_length * m_perp) / math.sqrt(1 + m_perp ** 2))

        if m < 0:
            x1 = x - (edge_length / math.sqrt(1 + m_perp ** 2))
            y1 = y - ((edge_length * m_perp) / math.sqrt(1 + m_perp ** 2))
            x2 = x + (edge_length / math.sqrt(1 + m_perp ** 2))
            y2 = y + ((edge_length * m_perp) / math.sqrt(1 + m_perp ** 2))

    return x1, y1, x2, y2