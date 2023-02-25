import { useCallback, useState } from 'react';

export function useModalStateValue() {
    const [isModalOpen, setModalOpen] = useState(false);

    return {
        closeModal: useCallback(() => setModalOpen(false), []),
        isModalOpen,
        openModal: useCallback(() => setModalOpen(true), []),
    };
}
