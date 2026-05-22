import os
from functools import wraps
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session, declarative_base

# 配置数据库连接URL，请根据实际情况修改或从环境变量获取
# 格式: postgresql://[user]:[password]@[host]:[port]/[dbname]


from db_conf import DB2


DATABASE_URL = os.environ.get(
    "DATABASE_URL", 
    DB2
)

# 创建引擎
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# 创建Session工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 创建声明性基类（用于定义ORM模型）
Base = declarative_base()

def with_session(func):
    """
    自动管理数据库 Session 的装饰器。
    它会自动创建 session 并作为关键字参数传入目标函数。
    函数执行成功后自动 commit，发生异常时自动 rollback，最后自动 close。
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 如果调用者已经传入了 session，则直接使用传入的 session (方便嵌套调用事务)
        if 'session' in kwargs and kwargs['session'] is not None:
            return func(*args, **kwargs)
            
        # 否则创建新的 session
        session: Session = SessionLocal()
        try:
            kwargs['session'] = session
            result = func(*args, **kwargs)
            session.commit()
            return result
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
            
    return wrapper

# ----- 使用示例 -----
#
# from sqlalchemy.sql import text

# @with_session
# def test(session: Session = None):
#     ret = session.execute(text("SELECT * from users"))
#     print(ret.fetchall())

# test()