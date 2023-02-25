import CheckoutModal from '@/components/SellOutCheckOut/CheckoutModal';
import { createContext, ReactNode, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import { Context } from 'vm';

const attr = 'data-rk';

const ThemeIdContext = createContext<string | undefined>(undefined);
const createThemeRootProps = (id: string | undefined) => ({ [attr]: id || '' });

export const useThemeRootProps = () => {
    const id = useContext(ThemeIdContext);
    return createThemeRootProps(id);
};

function useModalStateValue() {
    const [isModalOpen, setModalOpen] = useState(false);
    const [modalType, setModalType] = useState<'NFT' | 'PRODUCT'>('PRODUCT');
    const [modalTheme, setModalTheme] = useState<'LIGHT' | 'DARK'>('LIGHT');
    const [modalAccent, setModalAccent] = useState<'BLUE' | 'PURPLE' | 'PINK' | 'RED' | 'ORANGE' | 'GREEN'>('BLUE');

    return {
        closeModal: useCallback(() => setModalOpen(false), []),
        isModalOpen,
        openModal: useCallback(() => setModalOpen(true), []),
        setModalType,
        modalType,
        setModalTheme,
        modalTheme,
        setModalAccent,
        modalAccent,
    };
}

type ModalThemeTypes = 'LIGHT' | 'DARK' | 'SLATE';
type ModalAccentTypes = 'BLUE' | 'PURPLE' | 'PINK' | 'RED' | 'ORANGE' | 'GREEN';

interface ModalContextValue {
    sellOutModalOpen: boolean;
    openSellOutModal?: () => void;
    modalType: 'NFT' | 'PRODUCT';
    setModalType: (type: 'NFT' | 'PRODUCT') => void;
    modalTheme: ModalThemeTypes;
    setModalTheme: (theme: ModalThemeTypes) => void;
    modalAccent: ModalAccentTypes;
    setModalAccent: (accent: ModalAccentTypes) => void;
}

const ModalContext = createContext<ModalContextValue>({
    sellOutModalOpen: false,
    modalType: 'PRODUCT',
    setModalType: () => {},
    modalTheme: 'LIGHT',
    setModalTheme: () => {},
    modalAccent: 'BLUE',
    setModalAccent: () => {},
});

interface ModalProviderProps {
    children: ReactNode;
}

export function ModalProvider({ children }: ModalProviderProps) {
    const {
        closeModal,
        isModalOpen: sellOutModalOpen,
        openModal: openSellOutModal,
        setModalType,
        modalType,
        setModalTheme,
        modalTheme,
        setModalAccent,
        modalAccent,
    } = useModalStateValue();

    return (
        <ModalContext.Provider
            value={useMemo(
        () => ({
        closeModal,
        sellOutModalOpen,
        openSellOutModal,
        setModalType,
        modalType,
        setModalTheme,
        modalTheme,
        setModalAccent,
        modalAccent,
    }),
        [
            closeModal,
            sellOutModalOpen,
            openSellOutModal,
            setModalType,
            modalType,
            modalTheme,
            setModalTheme,
            modalAccent,
            setModalAccent,
        ],
)}
>
    {children}
    {/* <CheckoutModal open={sellOutModalOpen} onClose={closeModal} /> */}
    </ModalContext.Provider>
);
}

export function useSelloutModal(type?: 'NFT' | 'PRODUCT') {
    const {
        sellOutModalOpen,
        openSellOutModal,
        closeModal,
        modalType,
        setModalType,
        modalTheme,
        setModalTheme,
        modalAccent,
        setModalAccent,
    } = useContext(ModalContext);
    return {
        sellOutModalOpen,
        openSellOutModal,
        closeModal,
        modalType,
        setModalType,
        modalTheme,
        setModalTheme,
        setModalAccent,
        modalAccent,
    };
}
